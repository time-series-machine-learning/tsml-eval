"""Build an HC2 result from native GEAR component train/test files.

The script discovers every dataset/resample pair complete in all four
component directories, applies standard HC2 accuracy-to-the-fourth weights,
and writes combined train and test files. It defaults to the GEAR-Comp family
but can also combine a shared GEAR-Auto representation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

from tsml_eval.evaluation.storage import ClassifierResults
from tsml_eval.utils.functions import time_to_milliseconds
from tsml_eval.utils.results_writing import write_classification_results


COMPONENTS = ("Arsenal", "DrCIF", "STC", "TDE")
SOURCE_PREFIX = "GEAR-Comp-Native"
RESULT_NAME = "GEAR-Comp-HC2"


def _load(path: Path) -> ClassifierResults:
    return ClassifierResults().load_from_file(str(path), verify_values=True)


def _identifies_source(path: Path, expected_name: str, split: str) -> bool:
    """Return whether a result header identifies the expected source run."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open(encoding="utf-8") as file:
            first_line = file.readline().strip()
    except (OSError, UnicodeError):
        return False
    return (
        f",{expected_name} (" in first_line
        and f"),{split.upper()}," in first_line
    )


def _source_file(
    results_root: Path,
    source_prefix: str,
    component: str,
    dataset: str,
    split: str,
    resample_id: int,
) -> Path:
    return (
        results_root
        / f"{source_prefix}-{component}"
        / "Predictions"
        / dataset
        / f"{split}Resample{resample_id}.csv"
    )


def _discover_complete_keys(
    results_root: Path, source_prefix: str
) -> list[tuple[str, int]]:
    keys = None
    for component in COMPONENTS:
        component_root = results_root / f"{source_prefix}-{component}" / "Predictions"
        component_keys = set()
        if component_root.is_dir():
            for train_file in component_root.glob("*/trainResample*.csv"):
                suffix = train_file.stem.removeprefix("trainResample")
                if not suffix.isdigit():
                    continue
                test_file = train_file.with_name(f"testResample{suffix}.csv")
                expected_name = f"{source_prefix}-{component}"
                if _identifies_source(
                    train_file, expected_name, "TRAIN"
                ) and _identifies_source(test_file, expected_name, "TEST"):
                    component_keys.add((train_file.parent.name, int(suffix)))
        keys = component_keys if keys is None else keys.intersection(component_keys)
    return sorted(keys or ())


def _validate_aligned(results: list[ClassifierResults], split: str):
    reference = results[0]
    for component, result in zip(COMPONENTS[1:], results[1:]):
        if result.n_cases != reference.n_cases:
            raise ValueError(f"{split} n_cases differs for {component}.")
        if result.n_classes != reference.n_classes:
            raise ValueError(f"{split} n_classes differs for {component}.")
        if not np.array_equal(result.class_labels, reference.class_labels):
            raise ValueError(f"{split} labels differ for {component}.")
    labels = np.asarray(reference.class_labels)
    rounded = np.rint(labels)
    if not np.allclose(labels, rounded):
        raise ValueError(f"{split} labels are not integer encoded.")
    return rounded.astype(int), reference.n_classes


def _combine_probabilities(
    results: list[ClassifierResults], weights: np.ndarray
) -> np.ndarray:
    probabilities = np.zeros_like(results[0].probabilities, dtype=float)
    for result, weight in zip(results, weights):
        probabilities += np.asarray(result.probabilities) * weight
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError("Combined probabilities contain a zero row sum.")
    return probabilities / totals


def _predictions(probabilities: np.ndarray, labels, random_state: int):
    classes = np.unique(labels)
    if len(classes) != probabilities.shape[1]:
        raise ValueError("Label classes do not match probability columns.")
    rng = check_random_state(random_state)
    return np.asarray(
        [
            classes[rng.choice(np.flatnonzero(row == row.max()))]
            for row in probabilities
        ]
    )


def _sum_time(results: list[ClassifierResults], attribute: str) -> float:
    values = []
    for result in results:
        value = getattr(result, attribute)
        if value is None or value < 0:
            return -1
        values.append(time_to_milliseconds(value, result.time_unit))
    return float(sum(values))


def _max_memory(results: list[ClassifierResults]) -> float:
    values = [result.memory_usage for result in results]
    usable = [value for value in values if value is not None and value >= 0]
    return float(max(usable)) if usable else -1


def build_one(
    results_root: Path,
    dataset: str,
    resample_id: int,
    overwrite: bool,
    source_prefix: str = SOURCE_PREFIX,
    result_name: str = RESULT_NAME,
) -> str:
    output_dir = results_root / result_name / "Predictions" / dataset
    output_train = output_dir / f"trainResample{resample_id}.csv"
    output_test = output_dir / f"testResample{resample_id}.csv"
    if not overwrite and all(
        path.is_file() and path.stat().st_size > 0
        for path in (output_train, output_test)
    ):
        return "skipped"

    for component in COMPONENTS:
        expected_name = f"{source_prefix}-{component}"
        for split in ("train", "test"):
            source_file = _source_file(
                results_root,
                source_prefix,
                component,
                dataset,
                split,
                resample_id,
            )
            if not _identifies_source(source_file, expected_name, split):
                raise ValueError(
                    f"{source_file} is not a {expected_name} "
                    f"{split.upper()} result."
                )

    train_results = [
        _load(
            _source_file(
                results_root,
                source_prefix,
                component,
                dataset,
                "train",
                resample_id,
            )
        )
        for component in COMPONENTS
    ]
    test_results = [
        _load(
            _source_file(
                results_root,
                source_prefix,
                component,
                dataset,
                "test",
                resample_id,
            )
        )
        for component in COMPONENTS
    ]
    train_labels, n_classes = _validate_aligned(train_results, "TRAIN")
    test_labels, test_n_classes = _validate_aligned(test_results, "TEST")
    if test_n_classes != n_classes:
        raise ValueError("TRAIN and TEST contain different numbers of classes.")

    component_accuracies = np.asarray(
        [result.accuracy for result in train_results], dtype=float
    )
    weights = component_accuracies**4
    train_probabilities = _combine_probabilities(train_results, weights)
    test_probabilities = _combine_probabilities(test_results, weights)
    train_predictions = _predictions(
        train_probabilities, train_labels, resample_id
    )
    test_predictions = _predictions(test_probabilities, test_labels, resample_id)

    fit_time = _sum_time(test_results, "fit_time")
    predict_time = _sum_time(test_results, "predict_time")
    fit_and_estimate_time = _sum_time(train_results, "fit_and_estimate_time")
    memory_usage = _max_memory(test_results)
    parameter_info = json.dumps(
        {
            "alpha": 4,
            "components": list(COMPONENTS),
            "source_prefix": source_prefix,
            "native_train_accuracies": dict(
                zip(COMPONENTS, component_accuracies.tolist())
            ),
            "weights": dict(zip(COMPONENTS, weights.tolist())),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    comment = (
        f"Constructed from {source_prefix} files using each HC2 component's "
        "native train-estimate mechanism and accuracy^4 weighting."
    )

    write_classification_results(
        train_predictions,
        train_probabilities,
        train_labels,
        result_name,
        dataset,
        str(results_root),
        full_path=False,
        first_line_classifier_name=f"{result_name} (FromNativeComponentFiles)",
        split="TRAIN",
        resample_id=resample_id,
        time_unit="MILLISECONDS",
        first_line_comment=comment,
        parameter_info=parameter_info,
        accuracy=accuracy_score(train_labels, train_predictions),
        fit_time=fit_time,
        predict_time=-1,
        benchmark_time=-1,
        memory_usage=memory_usage,
        n_classes=n_classes,
        train_estimate_method="HC2-NATIVE-COMPONENT-ESTIMATES",
        train_estimate_time=-1,
        fit_and_estimate_time=fit_and_estimate_time,
    )
    write_classification_results(
        test_predictions,
        test_probabilities,
        test_labels,
        result_name,
        dataset,
        str(results_root),
        full_path=False,
        first_line_classifier_name=f"{result_name} (FromNativeComponentFiles)",
        split="TEST",
        resample_id=resample_id,
        time_unit="MILLISECONDS",
        first_line_comment=comment,
        parameter_info=parameter_info,
        accuracy=accuracy_score(test_labels, test_predictions),
        fit_time=fit_time,
        predict_time=predict_time,
        benchmark_time=-1,
        memory_usage=memory_usage,
        n_classes=n_classes,
        train_estimate_method="N/A",
        train_estimate_time=-1,
        fit_and_estimate_time=fit_and_estimate_time,
    )
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--resample-id", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-prefix", default=SOURCE_PREFIX)
    parser.add_argument("--result-name", default=RESULT_NAME)
    args = parser.parse_args()

    keys = _discover_complete_keys(args.results_root, args.source_prefix)
    if args.dataset:
        keys = [key for key in keys if key[0] in set(args.dataset)]
    if args.resample_id is not None:
        keys = [key for key in keys if key[1] == args.resample_id]

    written = skipped = 0
    for dataset, resample_id in keys:
        status = build_one(
            args.results_root,
            dataset,
            resample_id,
            overwrite=args.overwrite,
            source_prefix=args.source_prefix,
            result_name=args.result_name,
        )
        if status == "written":
            written += 1
            print(f"WROTE {args.result_name}/{dataset}/resample{resample_id}")
        else:
            skipped += 1
    print(f"Complete source cells: {len(keys)}; written: {written}; skipped: {skipped}")


if __name__ == "__main__":
    main()
