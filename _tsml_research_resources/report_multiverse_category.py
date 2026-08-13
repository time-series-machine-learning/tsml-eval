"""Report Multiverse completion and running jobs for one classifier category."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from _tsml_research_resources import multiverse_controller as controller  # noqa: E402


def _normalise(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _find_category(config, requested):
    wanted = _normalise(requested)
    matches = [
        category
        for category in config.categories
        if _normalise(category.name) == wanted
    ]
    if not matches:
        available = ", ".join(category.name for category in config.categories)
        raise ValueError(f"Unknown category {requested!r}. Available: {available}")
    return matches[0]


def _classifier_rows(config, category, datasets, snapshot):
    total = len(datasets) * config.resamples
    rows = []
    for classifier in category.classifiers:
        single = controller.Category(category.name, (classifier,))
        complete = controller._count_complete(config, single, datasets)
        running = sum(
            snapshot.states.get(task.job_key) == "RUNNING"
            for task in controller._iter_tasks(single, datasets, config.resamples)
        )
        rows.append((classifier, complete, total, running))
    return rows


def _compose_report(config, category, datasets, snapshot):
    rows = _classifier_rows(config, category, datasets, snapshot)
    complete = sum(row[1] for row in rows)
    total = sum(row[2] for row in rows)
    running = sum(row[3] for row in rows)
    percent = 100 * complete / total if total else 100.0

    lines = [
        f"Updated:  {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Category: {category.name}",
        f"Complete: {complete}/{total} ({percent:.1f}%)",
        f"Running:  {running}",
    ]
    if snapshot.error:
        lines.append(f"Slurm:    {snapshot.error}")
    lines.extend(
        (
            "",
            (
                f"{'Classifier':<26} {'Complete':>10} {'Total':>8} "
                f"{'Progress':>9} {'Running':>9}"
            ),
            "-" * 68,
        )
    )
    for classifier, done, expected, active in rows:
        progress = 100 * done / expected if expected else 100.0
        lines.append(
            f"{classifier:<26} {done:>10} {expected:>8} "
            f"{progress:>8.1f}% {active:>9}"
        )
    return "\n".join(lines) + "\n"


def _parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "category",
        nargs="?",
        default="IntervalBased",
        help="Category to report; spelling is case/separator insensitive.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=script_dir / "multiverse_controller.toml",
        help="Controller TOML providing paths, resamples, and classifiers.",
    )
    return parser.parse_args()


def main():
    """Load cluster state and print one category's current progress."""
    args = _parse_args()
    config = controller._load_config(args.config)
    category = _find_category(config, args.category)
    # Report the full archive even when submission configuration defers datasets.
    datasets = controller._read_datasets(config.dataset_file)
    snapshot = controller._query_slurm(config)
    print(_compose_report(config, category, datasets, snapshot), end="")  # noqa: T201


if __name__ == "__main__":
    main()
