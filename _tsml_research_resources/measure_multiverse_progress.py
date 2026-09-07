"""Report progress for MultiverseCore fold-zero classification experiments."""

import argparse
import getpass
import re
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from tsml_eval.utils.results_validation import validate_results_file


DEFAULT_CLASSIFIERS = (
    "CIF",
    "HC2",
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


def _latest_log_text(output_dir, classifier, dataset):
    log_dir = output_dir / classifier / dataset
    try:
        log_files = list(log_dir.glob("*.out")) + list(log_dir.glob("*.err"))
    except OSError:
        return None
    if not log_files:
        return None

    attempts = {}
    for log_file in log_files:
        attempts.setdefault(log_file.stem, []).append(log_file)

    try:
        latest = max(
            attempts.values(),
            key=lambda files: max(file.stat().st_mtime for file in files),
        )
    except OSError:
        return None

    contents = []
    for log_file in latest:
        try:
            contents.append(
                log_file.read_text(encoding="utf-8", errors="replace")
            )
        except OSError:
            pass
    return "\n".join(contents) if contents else None


def _failure_reason(output_dir, classifier, dataset):
    text = _latest_log_text(output_dir, classifier, dataset)
    if text is None:
        return "No logs"

    lower = text.lower()
    if re.search(r"oom[_ -]?kill|out[ -]?of[ -]?memory", lower):
        return "OOM"
    if "missing values" in lower and "cannot handle" in lower:
        return "Unsupported missing values"
    if "unequal length" in lower and "cannot handle" in lower:
        return "Unsupported unequal length"
    if "n_timepoints must be >=" in lower:
        return "Series too short"
    if "smaller than the min shapelet length" in lower:
        return "Minimum shapelet length"
    if "due to time limit" in lower or "time limit exceeded" in lower:
        return "Time limit"
    if "cancelled" in lower:
        return "Cancelled"
    if "traceback (most recent call last)" in lower:
        return "Python exception"
    if re.search(r"\bkilled\b", lower):
        return "Killed"
    if re.search(r"(^|\n).*error:", lower):
        return "Slurm/runtime error"
    return "No terminal status"


def _progress_rows(
    results_dir, output_dir, classifiers, datasets, fold, active_jobs
):
    rows = []
    failures = []
    filename = f"testResample{fold}.csv"

    for classifier in classifiers:
        complete = 0
        invalid = 0
        running = 0
        pending = 0
        oom = 0
        failed = 0
        unknown = 0

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
                    continue
                elif any(state == "PENDING" for state in states):
                    pending += 1
                    continue

            reason = _failure_reason(output_dir, classifier, dataset)
            failures.append((classifier, dataset, reason))
            if reason == "OOM":
                oom += 1
            elif reason in {"No logs", "No terminal status"}:
                unknown += 1
            else:
                failed += 1

        rows.append(
            (
                classifier,
                complete,
                invalid,
                running,
                pending,
                oom,
                failed,
                unknown,
                len(datasets),
            )
        )

    return rows, failures


def _print_report(
    results_dir,
    output_dir,
    dataset_file,
    rows,
    failures,
    active_jobs,
    show_failures,
):
    print(f"Updated:      {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"Results:      {results_dir}")
    print(f"Output logs:  {output_dir}")
    print(f"Dataset list: {dataset_file}")
    print()
    print(
        f"{'Classifier':<18} {'Complete':>8} {'Invalid':>7} "
        f"{'Running':>7} {'Pending':>7} {'OOM':>5} {'Failed':>6} "
        f"{'Unknown':>7} {'Progress':>9}"
    )
    print("-" * 101)

    totals = [0] * 8
    for row in rows:
        (
            classifier,
            complete,
            invalid,
            running,
            pending,
            oom,
            failed,
            unknown,
            total,
        ) = row
        progress = 100 * complete / total if total else 100.0
        print(
            f"{classifier:<18} {complete:>8} {invalid:>7} "
            f"{running:>7} {pending:>7} {oom:>5} {failed:>6} "
            f"{unknown:>7} {progress:>8.1f}%"
        )
        for index, value in enumerate(
            (complete, invalid, running, pending, oom, failed, unknown, total)
        ):
            totals[index] += value

    progress = 100 * totals[0] / totals[7] if totals[7] else 100.0
    print("-" * 101)
    print(
        f"{'TOTAL':<18} {totals[0]:>8} {totals[1]:>7} "
        f"{totals[2]:>7} {totals[3]:>7} {totals[4]:>5} {totals[5]:>6} "
        f"{totals[6]:>7} {progress:>8.1f}%"
    )

    if failures:
        print("\nInactive missing-result diagnoses:")
        reasons = Counter(reason for _, _, reason in failures)
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count}")

    if show_failures and failures:
        print("\nInactive missing-result details:")
        for classifier, dataset, reason in failures:
            print(f"  {classifier}/{dataset}: {reason}")

    if active_jobs is None:
        print(
            "\nSlurm status unavailable; live jobs may appear as No terminal status."
        )


def _write_hc2_oom_list(output_file, failures):
    datasets = [
        dataset
        for classifier, dataset, reason in failures
        if classifier == "HC2" and reason == "OOM"
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="\n") as file:
        if datasets:
            file.write("\n".join(datasets) + "\n")
    return len(datasets)


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Slurm output root; defaults to <results-dir>/output.",
    )
    parser.add_argument(
        "--oom-hc2-file",
        type=Path,
        help="HC2 OOM list; defaults to oom_hc2.txt beside the dataset list.",
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
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="List every inactive classifier/dataset pair and its diagnosis.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    datasets = _read_datasets(args.dataset_file)
    output_dir = args.output_dir or args.results_dir / "output"
    oom_hc2_file = args.oom_hc2_file or args.dataset_file.with_name("oom_hc2.txt")

    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        active_jobs = _active_jobs(getpass.getuser())
        rows, failures = _progress_rows(
            args.results_dir,
            output_dir,
            args.classifiers,
            datasets,
            args.fold,
            active_jobs,
        )
        oom_count = _write_hc2_oom_list(oom_hc2_file, failures)
        _print_report(
            args.results_dir,
            output_dir,
            args.dataset_file,
            rows,
            failures,
            active_jobs,
            args.show_failures,
        )
        print(f"\nHC2 OOM list: {oom_hc2_file} ({oom_count} problems)")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
