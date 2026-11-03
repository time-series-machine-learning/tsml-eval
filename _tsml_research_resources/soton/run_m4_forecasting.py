"""Run a forecaster over the full M4 dataset on a single multi-core machine.

Equivalent to calling remote_forecasting_experiments.py once per series with
``-kw fixed_horizon true bool -kw season_length <period> int`` (the setup used for the
earlier M4 runs), but each .tsf file is loaded once in the parent process and shared
with forked workers, instead of being re-parsed for every one of the 100k series.

Results are written by tsml-eval's ``run_forecasting_experiment`` so the files are in
the standard format at
``<results_dir>/<forecaster_name>/Predictions/<dataset>_<series>/testResample0.csv``.
Existing result files are skipped, so the script can be re-run to resume.

Usage
-----
python run_m4_forecasting.py <data_dir> <results_dir> <forecaster> [options]

    --name NAME        name to write results under (default: forecaster)
    --n-jobs N         number of worker processes (default: cpu_count - 4)
    --datasets A,B     comma separated subset of M4 datasets (default: all six)
    --limit K          only run the first K series of each dataset (for testing)
    --timeout S        per-series time limit in seconds (default: 3600)
"""

import os

for _v in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[_v] = "1"
# run_forecasting_experiment polls memory every MEMRECORD_INTERVAL seconds (default 5)
# and cannot return before the first poll, which would dominate the ~0.1s fits here.
os.environ.setdefault("MEMRECORD_INTERVAL", "0.05")

import argparse
import multiprocessing as mp
import signal
import sys
import time
import traceback
import warnings

import numpy as np

M4_DATASETS = [
    "m4_yearly_dataset",
    "m4_quarterly_dataset",
    "m4_monthly_dataset",
    "m4_weekly_dataset",
    "m4_daily_dataset",
    "m4_hourly_dataset",
]

# Shared with forked workers (copy on write).
_SERIES = {}
_ARGS = None


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _load(data_dir, datasets, limit):
    from aeon.datasets import load_from_tsf_file

    from tsml_eval.experiments.download_datasets import _seasonal_period_for

    tasks = []
    for dataset in datasets:
        df, metadata = load_from_tsf_file(f"{data_dir}/{dataset}/{dataset}.tsf")
        horizon = int(metadata["forecast_horizon"])
        period = _seasonal_period_for(dataset, metadata)
        names = df["series_name"].tolist()
        values = df["series_value"].tolist()
        if limit is not None:
            names, values = names[:limit], values[:limit]
        for name, value in zip(names, values):
            key = f"{dataset}_{name}"
            _SERIES[key] = (np.asarray(value, dtype=float), horizon, period)
            tasks.append(key)
        print(
            f"{dataset}: {len(names)} series, horizon={horizon}, "
            f"seasonal_period={period}",
            flush=True,
        )
    return tasks


def _run_one(key):
    from tsml_eval.experiments._get_forecaster import get_forecaster_by_name
    from tsml_eval.experiments.experiments import run_forecasting_experiment

    args = _ARGS
    out = f"{args.results_dir}/{args.name}/Predictions/{key}/testResample0.csv"
    if os.path.exists(out):
        return key, "skip", 0.0, ""

    series, horizon, period = _SERIES[key]
    start = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(args.timeout)
    try:
        if not np.isfinite(series).all():
            raise ValueError("series contains non-finite values")
        if len(series) <= horizon:
            raise ValueError(f"series length {len(series)} <= horizon {horizon}")
        # Same split as load_and_run_remote_forecasting_experiment(fixed_horizon=True)
        train = series[:-horizon]
        test = series[-horizon:]
        forecaster = get_forecaster_by_name(
            args.forecaster, random_state=0, n_jobs=1, season_length=period
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_forecasting_experiment(
                train,
                test,
                forecaster,
                args.results_dir,
                forecaster_name=args.name,
                dataset_name=key,
                random_seed=0,
                benchmark_time=True,
                prediction_horizon=horizon,
                forecasting_method="auto",
            )
        status, msg = "ok", ""
    except _Timeout:
        status, msg = "timeout", f"exceeded {args.timeout}s"
    except Exception:
        status, msg = "error", traceback.format_exc()
    finally:
        signal.alarm(0)
    return key, status, time.time() - start, msg


def _init_worker(benchmark):
    # The hardware benchmark is a property of the machine, not the series. Compute it
    # once (in the parent) rather than spending ~2s per series re-running it.
    import tsml_eval.experiments.experiments as ex

    ex.timing_benchmark = lambda *a, **k: benchmark
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    global _ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("results_dir")
    parser.add_argument("forecaster")
    parser.add_argument("--name", default=None)
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 4))
    parser.add_argument("--datasets", default=",".join(M4_DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    args.name = args.name or args.forecaster
    args.results_dir = args.results_dir.rstrip("/")
    _ARGS = args

    tasks = _load(args.data_dir, args.datasets.split(","), args.limit)
    todo = [
        k
        for k in tasks
        if not os.path.exists(
            f"{args.results_dir}/{args.name}/Predictions/{k}/testResample0.csv"
        )
    ]
    print(f"{len(tasks)} series, {len(todo)} to run, {args.n_jobs} workers", flush=True)

    from tsml_eval.experiments._get_forecaster import get_forecaster_by_name
    from tsml_eval.utils.experiments import timing_benchmark

    benchmark = timing_benchmark(random_state=0)
    print(f"timing benchmark: {benchmark}", flush=True)

    # Warm up numba in the parent so forked workers inherit compiled functions.
    if todo:
        series, horizon, period = _SERIES[todo[0]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            get_forecaster_by_name(
                args.forecaster, random_state=0, n_jobs=1, season_length=period
            ).iterative_forecast(series[:-horizon], horizon)

    # Longest series first so stragglers do not hold up the end of the run.
    todo.sort(key=lambda k: -len(_SERIES[k][0]))

    log_dir = f"{args.results_dir}/{args.name}"
    os.makedirs(log_dir, exist_ok=True)
    fail_log = open(f"{log_dir}/failures.log", "a")

    counts = {"ok": 0, "skip": 0, "error": 0, "timeout": 0}
    start = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.n_jobs, initializer=_init_worker, initargs=(benchmark,)) as pool:
        for i, (key, status, secs, msg) in enumerate(
            pool.imap_unordered(_run_one, todo, chunksize=4), 1
        ):
            counts[status] += 1
            if status in ("error", "timeout"):
                fail_log.write(f"=== {key} {status} ({secs:.1f}s)\n{msg}\n")
                fail_log.flush()
            if i % 1000 == 0 or i == len(todo) or status != "ok":
                el = time.time() - start
                eta = el / i * (len(todo) - i)
                print(
                    f"[{i}/{len(todo)}] {counts} elapsed={el / 60:.1f}m "
                    f"eta={eta / 60:.1f}m"
                    + (f" last={key} {status}" if status != "ok" else ""),
                    flush=True,
                )
    fail_log.close()
    print(f"Done in {(time.time() - start) / 60:.1f}m: {counts}", flush=True)
    return 0 if counts["error"] + counts["timeout"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
