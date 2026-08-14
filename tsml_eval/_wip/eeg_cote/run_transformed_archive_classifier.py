"""Run an archive classifier on a previously persisted EEG transform."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from aeon.datasets import load_from_ts_file

from tsml_eval.experiments._get_classifier import _make_hc2_or_component
from tsml_eval.experiments.experiments import run_classification_experiment


CLASSIFIER_COMPONENT_NAMES = {
    "HC2": "hc2",
    "Arsenal": "arsenal",
    "DrCIF": "drcif",
    "STC": "stc",
    "TDE": "tde",
}


def _required_transform_files(
    transform_root: Path, transform_name: str, problem: str
) -> tuple[Path, Path, Path]:
    directory = transform_root / transform_name / problem
    files = (
        directory / f"{problem}_TRAIN.ts",
        directory / f"{problem}_TEST.ts",
        directory / "transform_summary.json",
    )
    missing = [path for path in files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "The saved transform is incomplete:\n" + "\n".join(map(str, missing))
        )
    return files


def run_transformed_classifier(
    *,
    transform_root: Path,
    results_root: Path,
    transform_name: str,
    problem: str,
    classifier_name: str,
    random_state: int,
    build_train_file: bool,
) -> None:
    """Load a persisted representation and run one untransformed classifier."""
    train_file, test_file, summary_file = _required_transform_files(
        transform_root, transform_name, problem
    )
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary.get("problem") != problem:
        raise ValueError(f"Transform summary problem does not match {problem!r}.")
    if summary.get("transform_name") != transform_name:
        raise ValueError(
            f"Transform summary does not describe {transform_name!r}."
        )

    result_name = f"{transform_name}-{classifier_name}"
    test_result = (
        results_root
        / result_name
        / "Predictions"
        / problem
        / f"testResample{random_state}.csv"
    )
    train_result = test_result.with_name(f"trainResample{random_state}.csv")
    if test_result.is_file() and test_result.stat().st_size > 0:
        if not build_train_file or (
            train_result.is_file() and train_result.stat().st_size > 0
        ):
            print(f"{result_name}/{problem}: complete result exists; skipping.")
            return

    print(f"{result_name}/{problem}: loading saved transform {train_file.parent}")
    X_train, y_train = load_from_ts_file(train_file, return_type="numpy3D")
    X_test, y_test = load_from_ts_file(test_file, return_type="numpy3D")

    component_name = CLASSIFIER_COMPONENT_NAMES[classifier_name]
    classifier = _make_hc2_or_component(
        component=component_name,
        random_state=random_state,
        n_jobs=1,
        fit_contract=0,
        kwargs={},
    )
    print(
        f"{result_name}/{problem}: fitting {type(classifier).__name__} on "
        f"TRAIN {X_train.shape}, TEST {X_test.shape}"
    )
    run_classification_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier,
        results_path=results_root,
        classifier_name=result_name,
        dataset_name=problem,
        resample_id=random_state,
        build_test_file=True,
        build_train_file=build_train_file,
        benchmark_time=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--transform", required=True, choices=("GEAR-Auto",))
    parser.add_argument("--problem", required=True)
    parser.add_argument(
        "--classifier", required=True, choices=tuple(CLASSIFIER_COMPONENT_NAMES)
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--train-file", action="store_true")
    args = parser.parse_args()

    run_transformed_classifier(
        transform_root=args.transform_root,
        results_root=args.results_root,
        transform_name=args.transform,
        problem=args.problem,
        classifier_name=args.classifier,
        random_state=args.random_state,
        build_train_file=args.train_file,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
