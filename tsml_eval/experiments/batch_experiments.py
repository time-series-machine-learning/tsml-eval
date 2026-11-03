"""Run many experiments sequentially in a single Python process.

The cluster task farm scripts normally launch one Python process per experiment. For
cheap experiments (e.g. statistical forecasters on the 100k M4 series) the per-process
start-up then dominates: importing aeon/tsml-eval, parsing the .tsf data file, numba
compilation/cache loading and the hardware timing benchmark can take far longer than
the fit itself. This script amortises those costs over a list of experiments:

- the experiment script (e.g. remote_forecasting_experiments.py) is imported once,
- .tsf files are parsed once per process (see ``_load_tsf_cached`` in experiments.py),
- numba functions are compiled/loaded on the first experiment and reused,
- the timing benchmark is run once per random seed per process, not per experiment.

Usage
-----
python batch_experiments.py <experiment_script.py> <task_file>

Each line of ``task_file`` is ``<output_file><TAB><arguments>``, where ``arguments``
are exactly the command line arguments the experiment script would normally be given
and ``output_file`` receives that experiment's stdout/stderr (as the ``> file 2>&1``
redirect did for a single experiment). Experiment failures are logged to their output
file and do not stop the batch; the exit code is non-zero if any experiment failed.
"""

import os

# Must be set before numpy etc. are imported, as in the individual experiment scripts.
for _var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MPI_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
]:
    os.environ.setdefault(_var, "1")

import contextlib
import importlib.util
import shlex
import sys
import time
import traceback


def _load_script(path):
    spec = importlib.util.spec_from_file_location("_batch_experiment_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_experiment"):
        raise ValueError(f"{path} has no run_experiment(args) function.")
    return module


def _cache_timing_benchmark():
    """Run the hardware timing benchmark once per seed rather than per experiment."""
    import tsml_eval.experiments.experiments as experiments

    original = experiments.timing_benchmark
    cache = {}

    def cached(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = original(*args, **kwargs)
        return cache[key]

    experiments.timing_benchmark = cached


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    script_path, task_file = argv

    with open(task_file) as f:
        tasks = [line.rstrip("\n").split("\t", 1) for line in f if line.strip()]

    module = _load_script(script_path)
    _cache_timing_benchmark()

    n_failed = 0
    start = time.time()
    print(f"Running {len(tasks)} experiments from {task_file}", flush=True)
    for i, (output_file, arguments) in enumerate(tasks, 1):
        task_start = time.time()
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as out:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                try:
                    module.run_experiment(shlex.split(arguments))
                    status = "ok"
                except Exception:
                    traceback.print_exc()
                    status = "FAILED"
                    n_failed += 1
        # The experiment scripts try to assign a GPU on every call when
        # CUDA_VISIBLE_DEVICES is unset (each attempt queries the driver). If the
        # first attempt found none, stop trying for the rest of the batch.
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(
            f"[{i}/{len(tasks)}] {status} {time.time() - task_start:.1f}s "
            f"{output_file}",
            flush=True,
        )

    print(
        f"Finished {len(tasks)} experiments in {time.time() - start:.1f}s, "
        f"{n_failed} failed",
        flush=True,
    )
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
