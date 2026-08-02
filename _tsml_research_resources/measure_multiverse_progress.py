"""Report progress for MultiverseCore fold-zero classification experiments."""

import argparse
import getpass
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from tsml_eval.utils.results_validation import validate_results_file


DEFAULT_CLASSIFIERS = (
    "CIF",
    "FreshPRINCE",
    "QUANT",
    "RDST",
    "1NN-DTW",
    "Catch22",
    "DrCIF",
    "H-InceptionTime",
    "LiteTIME",
    "STC",
)


def _read_datasets(dataset_file):
    with dataset_file.open(encoding="utf-8-sig") as file:
        datasets = [
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if len(datasets) != len(set(datasets)):
        raise ValueError(f"Duplicate dataset names found in {dataset_file}")
    return datasets


def _active_jobs(username):
    if shutil.which("squeue") is None:
        return None

    try:
        result = subprocess.run(
            [
                "squeue",
                "--noheader",
                f"--user={username}",
                "--states=RUNNING,PENDING",
                "--format=%200j|%T",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    jobs = {}
    for line in result.stdout.splitlines():
        name, state = (field.strip() for field in line.rsplit("|", maxsplit=1))
        jobs.setdefault(name, []).append(state)
    return jobs


def _is_valid_result(result_file):
    try:
        return validate_results_file(result_file)
    except (IndexError, OSError, UnicodeError, ValueError):
        return False


def _progress_rows(results_dir, classifiers, datasets, fold, active_jobs):
    rows = []
    filename = f"testResample{fold}.csv"

    for classifier in classifiers:
        complete = 0
        invalid = 0
        running = 0
        pending = 0

        for dataset in datasets:
            result_file = (
                results_dir
                / classifier
                / "Predictions"
                / dataset
                / filename
            )
            if result_file.is_file():
                if _is_valid_result(result_file):
                    complete += 1
                else:
                    invalid += 1
                continue

            if active_jobs is not None:
                states = active_jobs.get(f"{classifier}_{dataset}", ())
                if any(state == "RUNNING" for state in states):
                    running += 1
                elif any(state == "PENDING" for state in states):
                    pending += 1

        missing = len(datasets) - complete - invalid
        inactive = missing - running - pending
        rows.append(
            (classifier, complete, invalid, running, pending, inactive, len(datasets))
        )

    return rows


def _print_report(results_dir, dataset_file, rows, active_jobs):
    print(f"Updated:      {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"Results:      {results_dir}")
    print(f"Dataset list: {dataset_file}")
    print()
    print(
        f"{'Classifier':<18} {'Complete':>8} {'Invalid':>7} "
        f"{'Running':>7} {'Pending':>7} {'Inactive':>8} {'Progress':>9}"
    )
    print("-" * 80)

    totals = [0] * 6
    for classifier, complete, invalid, running, pending, inactive, total in rows:
        progress = 100 * complete / total if total else 100.0
        print(
            f"{classifier:<18} {complete:>8} {invalid:>7} "
            f"{running:>7} {pending:>7} {inactive:>8} {progress:>8.1f}%"
        )
        for index, value in enumerate(
            (complete, invalid, running, pending, inactive, total)
        ):
            totals[index] += value

    progress = 100 * totals[0] / totals[5] if totals[5] else 100.0
    print("-" * 80)
    print(
        f"{'TOTAL':<18} {totals[0]:>8} {totals[1]:>7} "
        f"{totals[2]:>7} {totals[3]:>7} {totals[4]:>8} {progress:>8.1f}%"
    )
    if active_jobs is None:
        print("\nSlurm status unavailable; Running and Pending are shown as zero.")


def _parse_args():
    username = getpass.getuser()
    local_path = Path("/gpfs/home") / username
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=local_path / "Results/Multiverse/TestOnly/MultiverseCore",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=local_path / "DataSetLists/MultiverseCore.txt",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--classifiers", nargs="+", default=list(DEFAULT_CLASSIFIERS)
    )
    parser.add_argument(
        "--watch",
        type=int,
        metavar="SECONDS",
        default=0,
        help="Refresh continuously at this interval; zero prints once.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    datasets = _read_datasets(args.dataset_file)

    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        active_jobs = _active_jobs(getpass.getuser())
        rows = _progress_rows(
            args.results_dir,
            args.classifiers,
            datasets,
            args.fold,
            active_jobs,
        )
        _print_report(args.results_dir, args.dataset_file, rows, active_jobs)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
