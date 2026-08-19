# Briefing: H-InceptionTime on this 4-GPU workstation

You are starting cold on a machine that has just been set up to contribute to the
Multiverse paper's experiments. Read this whole file before doing anything. Ask the
user (not the assistant that wrote this file — the human at the keyboard) the open
questions at the end before you write or run code.

## Context

The Multiverse paper benchmarks ~17 classifiers over a 133-dataset archive
(resample 0 for the full run). The work is split across machines:

- **Hali** (UEA Slurm cluster): the 15 non-deep classifiers on CPU
  (`_tsml_research_resources/start_multiverse_full_resample0_cpu.sh`), plus
  LITETime-MV on GPU.
- **IridisX** (Southampton Slurm cluster): H-InceptionTime on GPU, split across two
  Slurm partitions (`a100` and `swarm_a100`) via
  `_tsml_research_resources/start_multiverse_full_resample0_hinception_gpu_iridisx.sh`.
- **This machine**: a dedicated 4-GPU workstation, no Slurm. Your job is to get
  H-InceptionTime resample-0 running here too, in parallel across the 4 local GPUs,
  to add throughput alongside IridisX rather than instead of it.

## What is already set up on this machine

- `~/Code/tsml-eval-gpu`, a clone of
  `https://github.com/time-series-machine-learning/tsml-eval`, checked out on
  `ajb/gpu`.
- `~/Code/aeon`, an existing checkout on branch `ajb/hc2`, installed editable
  (`pip install --editable . --no-deps`) into the `tsml-eval-gpu` conda env.
- The `tsml-eval-gpu` conda env (Python 3.13), with `tensorflow[and-cuda]` installed
  and confirmed to see all 4 GPUs (`tf.config.list_physical_devices("GPU")`), and
  `tsml-eval-gpu` (this checkout) installed editable into it.

Verify all of this still holds before building anything:

```bash
conda activate tsml-eval-gpu
python -c "
import tsml_eval, aeon, tensorflow as tf
print('tsml_eval:', tsml_eval.__file__)
print('aeon:', aeon.__file__, aeon.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
"
```

`tsml_eval.__file__` must be under `~/Code/tsml-eval-gpu`, `aeon.__file__` must be
under `~/Code/aeon`, and 4 GPUs must be listed.

## The key difference from Hali/IridisX: no Slurm here

`_tsml_research_resources/multiverse_controller.py` is the shared engine behind every
`start_multiverse_*.sh` script referenced above. It is built entirely around
`sbatch`/`squeue` — it generates a Slurm batch script per classifier/dataset pair and
submits it as an array job. **None of that submission machinery applies here.** Do not
try to run `multiverse_controller.py` directly or "trick" it into thinking this is a
Slurm node.

What you do need to replicate is the actual experiment invocation it generates. Read
`_batch_script()` in `multiverse_controller.py` (around line 490–659) for the full
reference. The parts that matter for a local run:

**The CLI call** (this is the thing that actually runs the experiment):

```bash
python -u -m tsml_eval.experiments.classification_experiments \
    <data_dir> <category_results_dir> H-InceptionTime <dataset_name> <resample_id>
```

- `<data_dir>` — the Multiverse archive root (see "Data" below).
- `<category_results_dir>` — `<results_root>/DeepLearning` (H-InceptionTime's category
  in every existing config; keep it, other tooling that reads these results expects
  it).
- `<resample_id>` — `0` for every task in this pass (this is a resample-0-only run,
  matching every other machine's scope).

This writes results to
`<category_results_dir>/H-InceptionTime/Predictions/<dataset_name>/testResample0.csv`
(and a matching `trainResample0.csv` only if `-tr` is passed, which this pass does
not use). That exact layout matters: it is what
`tsml_eval.evaluation.multiple_estimator_evaluation.evaluate_classifiers_by_problem`
and every sync/audit script this session has built expect. Do not restructure it.

**Per-process environment**, one GPU per worker — since there's no Slurm GRES
allocation doing this for you, you must set `CUDA_VISIBLE_DEVICES` yourself, before
each worker process starts (it must be set before TensorFlow initializes, so as an
env var on process launch, not something set from inside an already-running
interpreter):

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m tsml_eval.experiments.classification_experiments ...
CUDA_VISIBLE_DEVICES=1 python -u -m tsml_eval.experiments.classification_experiments ...
CUDA_VISIBLE_DEVICES=2 python -u -m tsml_eval.experiments.classification_experiments ...
CUDA_VISIBLE_DEVICES=3 python -u -m tsml_eval.experiments.classification_experiments ...
```

Also set these in each worker's environment, matching what the Slurm scripts export
(prevents each process from spawning threads that fight the other 3 for CPU):

```
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
LOKY_MAX_CPU_COUNT=1
TF_NUM_INTEROP_THREADS=1
TF_NUM_INTRAOP_THREADS=1
PYTHONUNBUFFERED=1
TF_CPP_MIN_LOG_LEVEL=2
```

If TensorFlow fails to find CUDA libraries when run this way (it shouldn't, since you
already verified GPU visibility from an interactive shell in this same env, but the
Slurm script does this defensively), the fallback the Slurm script uses is adding the
pip-installed `nvidia-*` package `lib` directories to `LD_LIBRARY_PATH` — see the
`device_setup` block in `_batch_script()` for the exact discovery command if needed.

## What you need to build

A small local driver that:

1. Knows the list of datasets to run (see open questions below — do not assume the
   full 133-dataset list without checking what's already done elsewhere).
2. Skips anything already complete on this machine
   (`<category_results_dir>/H-InceptionTime/Predictions/<dataset>/testResample0.csv`
   exists and is non-empty — same check as `_is_complete()` in
   `multiverse_controller.py`, feel free to reuse that function directly by importing
   it rather than reimplementing).
3. Runs up to 4 datasets concurrently, one per GPU, moving on to the next dataset in
   the list as each GPU frees up (a simple worker-pool / queue is enough — this does
   not need Slurm's array-job sophistication, just correct GPU pinning and not
   double-running a GPU).
4. Logs enough that a stuck or crashed run is easy to spot (stdout/stderr per task,
   at minimum).

Keep it simple. This machine does not need retry/backoff, memory-tier escalation, or
any of the Slurm-specific plumbing in `multiverse_controller.py` — those exist to
manage a shared cluster queue, which does not apply to a dedicated 4-GPU box you
fully control. A straightforward Python or bash script with a 4-slot worker pool is
the right scope.

## Open questions — ask the user before writing code

1. **Where does the Multiverse data archive live on this machine?** It likely is not
   here yet. Each dataset needs both the base and "clean" (`_eq`/`_nmv`/`_eq_nmv`)
   variant `.ts` files under `<data_dir>/<dataset>/`, matching the layout on
   Hali/IridisX (see the availability-check logic in
   `start_multiverse_full_resample0_cpu.sh` or
   `start_multiverse_full_resample0_hinception_gpu_iridisx.sh` for the exact file
   pattern). Ask whether to copy it from an existing machine, and how much of the
   133-dataset list is actually needed here (see next question) before pulling
   everything.
2. **Which datasets should this machine take?** IridisX is already working through
   the full list (split across its two queues). Running the same datasets here too
   would work but duplicates GPU time for no benefit. Ask the user whether to:
   - take a disjoint slice of the list (coordinate with what's running on IridisX), or
   - just run everything not yet complete, treating this as extra throughput and
     accepting some overlap, or
   - focus on a specific subset (e.g. the largest/slowest datasets, to relieve
     pressure on IridisX 's swarm queue).
3. **Where should results end up in the long run?** This machine has no shared
   filesystem with Hali/IridisX or the user's Windows machine. Results here will need
   to be synced back manually at some point (the user has used `robocopy` for this
   between Windows drives before) — confirm the destination and method before
   assuming results just accumulate here indefinitely.
4. **`results_root` on this machine** — pick a path and confirm it with the user
   before starting (e.g. `~/Results/Multiverse`, matching the naming convention used
   elsewhere, even though the filesystem itself is local and separate).

Do not guess at 2 and 3 — running the wrong dataset slice or losing track of where
results need to end up wastes real GPU-hours on a run that's explicitly meant to be
unattended.
