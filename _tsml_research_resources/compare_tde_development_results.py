"""Compare resample-0 TDE development variants with baseline TDE results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from tsml_eval.evaluation.storage import load_classifier_results
from tsml_eval.utils.functions import time_to_milliseconds


def _files(classifier_dir: Path) -> dict[str, Path]:
    predictions = classifier_dir / "Predictions"
    if not predictions.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {predictions}")
    return {
        dataset.name: result
        for dataset in predictions.iterdir()
        if dataset.is_dir()
        and (result := dataset / "testResample0.csv").is_file()
    }


def _load(path: Path):
    result = load_classifier_results(path)
    return {
        "accuracy": result.accuracy,
        "balanced_accuracy": result.balanced_accuracy,
        "log_loss": result.log_loss,
        "fit_seconds": time_to_milliseconds(result.fit_time, result.time_unit) / 1000,
        "predict_seconds": time_to_milliseconds(
            result.predict_time, result.time_unit
        )
        / 1000,
        "memory_gib": result.memory_usage / 1024**3,
        "correct": int(np.sum(result.class_labels == result.predictions)),
        "n_cases": result.n_cases,
    }


def _paired_summary(name_a, files_a, name_b, files_b, datasets):
    rows = []
    invalid = []
    for dataset in sorted(datasets):
        try:
            a = _load(files_a[dataset])
            b = _load(files_b[dataset])
        except Exception as error:  # diagnostic should continue past a bad file
            invalid.append((dataset, str(error)))
            continue
        rows.append((dataset, a, b))

    print(f"\n{name_a} versus {name_b}: {len(rows)} matched valid datasets")
    if invalid:
        print(f"  Invalid pairs excluded: {len(invalid)}")
        for dataset, error in invalid:
            print(f"    {dataset}: {error}")
    if not rows:
        return

    metrics = (
        ("accuracy", "mean"),
        ("balanced_accuracy", "mean"),
        ("log_loss", "mean"),
        ("fit_seconds", "median"),
        ("predict_seconds", "median"),
        ("memory_gib", "median"),
    )
    for metric, reduction in metrics:
        a_values = np.asarray([row[1][metric] for row in rows])
        b_values = np.asarray([row[2][metric] for row in rows])
        function = np.mean if reduction == "mean" else np.median
        print(
            f"  {metric} ({reduction}): {name_a}={function(a_values):.6g}, "
            f"{name_b}={function(b_values):.6g}, "
            f"difference={function(a_values - b_values):+.6g}"
        )

    accuracy_a = np.asarray([row[1]["accuracy"] for row in rows])
    accuracy_b = np.asarray([row[2]["accuracy"] for row in rows])
    differences = accuracy_a - accuracy_b
    wins = int(np.sum(differences > 1e-12))
    draws = int(np.sum(np.abs(differences) <= 1e-12))
    losses = int(np.sum(differences < -1e-12))
    p_value = 1.0 if np.allclose(differences, 0) else wilcoxon(differences).pvalue
    correct_difference = sum(row[1]["correct"] - row[2]["correct"] for row in rows)
    total_cases = sum(row[1]["n_cases"] for row in rows)
    fit_a = sum(row[1]["fit_seconds"] for row in rows)
    fit_b = sum(row[2]["fit_seconds"] for row in rows)
    print(f"  Accuracy wins/draws/losses: {wins}/{draws}/{losses}")
    print(f"  Paired Wilcoxon accuracy p-value: {p_value:.6g}")
    print(f"  Additional correct predictions: {correct_difference:+d}/{total_cases}")
    print(
        f"  Aggregate fit time: {name_a}={fit_a / 3600:.3f} h, "
        f"{name_b}={fit_b / 3600:.3f} h, ratio={fit_a / fit_b:.3f}"
    )

    ordered = sorted(rows, key=lambda row: row[1]["accuracy"] - row[2]["accuracy"])
    print("  Largest accuracy improvements:")
    for dataset, a, b in reversed(ordered[-5:]):
        print(
            f"    {dataset}: {a['accuracy'] - b['accuracy']:+.4f} "
            f"({a['accuracy']:.4f} vs {b['accuracy']:.4f})"
        )
    print("  Largest accuracy regressions:")
    for dataset, a, b in ordered[:5]:
        print(
            f"    {dataset}: {a['accuracy'] - b['accuracy']:+.4f} "
            f"({a['accuracy']:.4f} vs {b['accuracy']:.4f})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(r"C:\Temp"))
    args = parser.parse_args()

    names = (
        "TDE",
        "TDE_Dev",
        "TDE_Dev2",
        "TDE_Dev3",
        "TDE_Dev3-Uniform",
    )
    development_names = names[1:]
    result_files = {name: _files(args.root / name) for name in names}
    for name in names:
        print(f"{name}: {len(result_files[name])} resample-0 files")

    common = set.intersection(*(set(result_files[name]) for name in names))

    for name in development_names:
        pairs = set(result_files["TDE"]) & set(result_files[name])
        _paired_summary(
            name,
            result_files[name],
            "TDE",
            result_files["TDE"],
            pairs,
        )

    print(f"\nFair {len(names)}-way subset: {len(common)} datasets")
    for name in development_names:
        _paired_summary(
            name,
            result_files[name],
            "TDE",
            result_files["TDE"],
            common,
        )

    _paired_summary(
        "TDE_Dev3",
        result_files["TDE_Dev3"],
        "TDE_Dev",
        result_files["TDE_Dev"],
        common,
    )
    _paired_summary(
        "TDE_Dev3-Uniform",
        result_files["TDE_Dev3-Uniform"],
        "TDE_Dev3",
        result_files["TDE_Dev3"],
        common,
    )
    _paired_summary(
        "TDE_Dev3",
        result_files["TDE_Dev3"],
        "TDE_Dev2",
        result_files["TDE_Dev2"],
        common,
    )


if __name__ == "__main__":
    main()
