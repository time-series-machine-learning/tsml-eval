"""Train level-1 (stacking) models on the stacking problems, writing tsml-eval results.

    <results>/<archive>/<category>/<model>/Predictions/<dataset>/testResample<r>.csv

Classical models (default), or the tabular foundation models with --foundation:

    python run_stacking.py UCR --data ~/Data/Stacking --results ~/Results/Stacking
    python run_stacking.py UCR --foundation --jobs 1 --device cuda
    python run_stacking.py EEG --models LR CAWPE --datasets FibroUEA

Existing results files are skipped, so a stopped run can be restarted.
"""

import argparse
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path


def run_dataset(args):
    """Every model and resample of one dataset; returns (dataset, done, failed)."""
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(Path(__file__).parent))
    from stacking_models import stackers

    from tsml_eval.experiments import run_classification_experiment
    from tsml_eval.utils.datasets import load_experiment_data
    from tsml_eval.utils.experiments import _results_present

    data, results, archive, ds, n_res, models, foundation, device = args
    done = failed = 0
    for r in range(n_res):
        todo = {
            m: v
            for m, v in stackers(r, foundation=foundation, device=device).items()
            if (not models or m in models)
            and not _results_present(
                str(results / archive / v[0]), m, ds, resample_id=r, split="TEST"
            )
        }
        if not todo:
            continue
        try:
            X_tr, y_tr, X_te, y_te, _ = load_experiment_data(
                str(data / archive), ds, r, True
            )
        except Exception as e:  # noqa: BLE001
            print(f"LOAD FAIL {archive} {ds} r{r}: {str(e)[:80]}", flush=True)
            failed += len(todo)
            continue
        for m, (category, est) in todo.items():
            try:
                run_classification_experiment(
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    est,
                    str(results / archive / category),
                    classifier_name=m,
                    dataset_name=ds,
                    resample_id=r,
                    benchmark_time=False,
                )
                done += 1
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {archive} {m} {ds} r{r}: {str(e)[:200]}", flush=True)
                failed += 1
    return ds, done, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("archive", help="UCR, EEG or Multiverse")
    parser.add_argument("--data", default="C:/Results/Stacking/Data")
    parser.add_argument("--results", default="C:/Results/Stacking/Results")
    parser.add_argument("--models", nargs="+", help="names from stacking_models")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--foundation", action="store_true", help="run the foundation models instead"
    )
    parser.add_argument("--device", help="foundation models: cuda, cpu (default auto)")
    parser.add_argument("--jobs", type=int, default=8, help="parallel datasets")
    args = parser.parse_args()

    data = Path(args.data).expanduser()
    results = Path(args.results).expanduser()
    root = data / args.archive
    datasets = args.datasets or (root / "datasets.txt").read_text().split()
    n_res = int(
        (root / "components.txt").read_text().split("resamples:")[1].split()[0]
    )
    # largest datasets first so the long ones do not finish last
    datasets.sort(key=lambda d: -sum(f.stat().st_size for f in (root / d).iterdir()))
    jobs = [
        (data, results, args.archive, d, n_res, args.models, args.foundation,
         args.device)
        for d in datasets
    ]
    done = failed = 0
    with Pool(args.jobs) as pool:
        for ds, d, f in pool.imap_unordered(run_dataset, jobs):
            done, failed = done + d, failed + f
            print(f"{ds}: {d} written, {f} failed", flush=True)
    print(f"COMPLETE {args.archive}: {done} written, {failed} failed")


if __name__ == "__main__":
    main()
