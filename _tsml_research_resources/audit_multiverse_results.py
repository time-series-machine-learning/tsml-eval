#!/usr/bin/env python3
"""Audit Multiverse paper result completeness across datasets and resamples."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CORE_LIST = (
    SCRIPT_DIR
    / "dataset_lists"
    / "MultivariateClassification66-MultiverseMini.txt"
)
DEFAULT_FULL_LIST = (
    SCRIPT_DIR
    / "dataset_lists"
    / "Multivariate133Classification-MultiverseClean.txt"
)
RESULT_NAME = re.compile(r"^testResample(\d+)\.csv$")
CLEAN_SUFFIXES = ("_eq_nmv", "_eq", "_nmv")


@dataclass(frozen=True)
class Classifier:
    """Classifier and result-directory category."""

    category: str
    name: str


CLASSIFIERS = (
    Classifier("IntervalBased", "CIF-500"),
    Classifier("IntervalBased", "DrCIF-500"),
    Classifier("IntervalBased", "QUANT"),
    Classifier("DictionaryBased", "TDE"),
    Classifier("ShapeletBased", "RDST"),
    Classifier("ShapeletBased", "STC"),
    Classifier("ConvolutionBased", "Arsenal"),
    Classifier("ConvolutionBased", "MRHydra"),
    Classifier("ConvolutionBased", "ROCKET"),
    Classifier("DistanceBased", "1NN-DTW"),
    Classifier("FeatureBased", "Catch22"),
    Classifier("FeatureBased", "FreshPRINCE"),
    Classifier("Hybrid", "HC2"),
    Classifier("Hybrid", "RIST"),
    Classifier("DeepLearning", "H-InceptionTime"),
    Classifier("DeepLearning", "LITETime-MV"),
    Classifier("Other", "Dummy"),
)


def _base_name(name: str) -> str:
    for suffix in CLEAN_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _read_dataset_list(path: Path) -> tuple[str, ...]:
    values = tuple(
        _base_name(line.strip())
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Dataset list is empty or contains duplicates: {path}")
    return values


def _is_explicitly_excluded(dataset: str) -> bool:
    return (
        dataset == "LenDB"
        or dataset.startswith("DREAM")
        or dataset.startswith("S2Agri-")
    )


def _scan_results(results_root: Path):
    present = set()
    zero_length = []
    for classifier in CLASSIFIERS:
        predictions = (
            results_root / classifier.category / classifier.name / "Predictions"
        )
        if not predictions.is_dir():
            continue
        for dataset_dir in predictions.iterdir():
            if not dataset_dir.is_dir():
                continue
            for result_file in dataset_dir.glob("testResample*.csv"):
                match = RESULT_NAME.fullmatch(result_file.name)
                if match is None:
                    continue
                try:
                    size = result_file.stat().st_size
                except OSError:
                    size = 0
                key = (classifier.name, dataset_dir.name, int(match.group(1)))
                if size > 0:
                    present.add(key)
                else:
                    zero_length.append(result_file)
    return present, zero_length


def _classifier_rows(datasets, present, resamples):
    rows = []
    expected_files = len(datasets) * resamples
    for classifier in CLASSIFIERS:
        counts = {
            dataset: sum(
                (classifier.name, dataset, resample) in present
                for resample in range(resamples)
            )
            for dataset in datasets
        }
        complete_files = sum(counts.values())
        complete_datasets = sum(count == resamples for count in counts.values())
        rows.append(
            {
                "category": classifier.category,
                "classifier": classifier.name,
                "complete_files": complete_files,
                "expected_files": expected_files,
                "percent": 100 * complete_files / expected_files,
                "complete_datasets": complete_datasets,
                "total_datasets": len(datasets),
                "missing_files": expected_files - complete_files,
            }
        )
    return rows


def _dataset_rows(datasets, present, resamples):
    expected = len(CLASSIFIERS) * resamples
    rows = []
    for dataset in datasets:
        complete = sum(
            (classifier.name, dataset, resample) in present
            for classifier in CLASSIFIERS
            for resample in range(resamples)
        )
        rows.append(
            {
                "dataset": dataset,
                "complete_files": complete,
                "expected_files": expected,
                "percent": 100 * complete / expected,
                "complete": complete == expected,
            }
        )
    return rows


def _missing_rows(datasets, present, resamples, scope):
    rows = []
    for classifier in CLASSIFIERS:
        for dataset in datasets:
            missing = [
                resample
                for resample in range(resamples)
                if (classifier.name, dataset, resample) not in present
            ]
            if missing:
                rows.append(
                    {
                        "scope": scope,
                        "category": classifier.category,
                        "classifier": classifier.name,
                        "dataset": dataset,
                        "complete_resamples": resamples - len(missing),
                        "expected_resamples": resamples,
                        "missing_resamples": " ".join(map(str, missing)),
                    }
                )
    return rows


def _format_table(title, rows):
    lines = [title]
    lines.append(
        f"{'Classifier':<20} {'Files':>12} {'Percent':>9} "
        f"{'Complete datasets':>19} {'Missing':>10}"
    )
    lines.append("-" * 76)
    for row in rows:
        files = f"{row['complete_files']}/{row['expected_files']}"
        datasets = f"{row['complete_datasets']}/{row['total_datasets']}"
        lines.append(
            f"{row['classifier']:<20} {files:>12} {row['percent']:>8.1f}% "
            f"{datasets:>19} {row['missing_files']:>10}"
        )
    complete = sum(row["complete_files"] for row in rows)
    expected = sum(row["expected_files"] for row in rows)
    lines.append("-" * 76)
    lines.append(
        f"{'TOTAL':<20} {f'{complete}/{expected}':>12} "
        f"{100 * complete / expected:>8.1f}% {'':>19} {expected - complete:>10}"
    )
    return "\n".join(lines)


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--core-list", type=Path, default=DEFAULT_CORE_LIST)
    parser.add_argument("--full-list", type=Path, default=DEFAULT_FULL_LIST)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    results_root = args.results_root.resolve()
    if not results_root.is_dir():
        parser.error(f"Results root does not exist: {results_root}")

    core = _read_dataset_list(args.core_list)
    full = _read_dataset_list(args.full_list)
    eligible = tuple(dataset for dataset in full if not _is_explicitly_excluded(dataset))
    excluded = tuple(dataset for dataset in full if _is_explicitly_excluded(dataset))
    if not set(core).issubset(full):
        missing = sorted(set(core) - set(full))
        raise ValueError(f"Core datasets absent from full list: {missing}")

    print(f"Scanning: {results_root}", flush=True)
    present, zero_length = _scan_results(results_root)
    print(f"Non-empty paper result files found: {len(present):,}", flush=True)

    scopes = (
        ("core_30", core, 30),
        ("full_resample0", full, 1),
        ("full_30", full, 30),
        ("eligible_full_resample0", eligible, 1),
        ("eligible_full_30", eligible, 30),
    )
    classifier_reports = {
        name: _classifier_rows(datasets, present, resamples)
        for name, datasets, resamples in scopes
    }
    dataset_reports = {
        name: _dataset_rows(datasets, present, resamples)
        for name, datasets, resamples in scopes
    }

    sections = [
        f"Multiverse result audit - {datetime.now().isoformat(timespec='seconds')}",
        f"Results root: {results_root}",
        f"Paper classifiers: {len(CLASSIFIERS)}",
        f"Core datasets: {len(core)}",
        f"Full datasets: {len(full)}",
        f"Eligible full datasets: {len(eligible)}",
        f"Explicitly excluded ({len(excluded)}): {', '.join(excluded)}",
        f"Zero-length result files: {len(zero_length)}",
        "",
        _format_table(
            "1) Thirty resamples on Multiverse Core", classifier_reports["core_30"]
        ),
        "",
        _format_table(
            "2) Resample 0 on all Multiverse", classifier_reports["full_resample0"]
        ),
        "",
        _format_table(
            "2a) Resample 0 on eligible Multiverse (seven exclusions removed)",
            classifier_reports["eligible_full_resample0"],
        ),
        "",
        _format_table(
            "3) Thirty resamples on all Multiverse", classifier_reports["full_30"]
        ),
        "",
        _format_table(
            "3a) Thirty resamples on eligible Multiverse (seven exclusions removed)",
            classifier_reports["eligible_full_30"],
        ),
    ]

    for name, rows in dataset_reports.items():
        complete = sum(row["complete"] for row in rows)
        sections.append(
            f"Datasets fully complete across all 17 classifiers [{name}]: "
            f"{complete}/{len(rows)}"
        )

    report = "\n".join(sections) + "\n"
    print(report)

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "multiverse_audit.txt").write_text(
            report, encoding="utf-8", newline="\n"
        )
        for name, rows in classifier_reports.items():
            _write_csv(output_dir / f"{name}_by_classifier.csv", rows)
        for name, rows in dataset_reports.items():
            _write_csv(output_dir / f"{name}_by_dataset.csv", rows)
        missing = []
        for name, datasets, resamples in scopes[:3]:
            missing.extend(_missing_rows(datasets, present, resamples, name))
        _write_csv(output_dir / "missing_results.csv", missing)
        if zero_length:
            _write_csv(
                output_dir / "zero_length_results.csv",
                ({"path": str(path)} for path in zero_length),
            )
        metadata = {
            "results_root": str(results_root),
            "paper_classifiers": [classifier.name for classifier in CLASSIFIERS],
            "core_datasets": list(core),
            "full_datasets": list(full),
            "eligible_full_datasets": list(eligible),
            "explicitly_excluded": list(excluded),
        }
        (output_dir / "audit_definition.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Detailed audit files: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
