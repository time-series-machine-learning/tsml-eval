"""Run one OpenCloseFist LOSO GEAR component with its native train estimate."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from tsml_eval._wip.eeg_cote.run_native_gear_component import (
    COMPONENTS,
    NATIVE_METHODS,
    GEARNativeComponentPipeline,
)
from tsml_eval._wip.eeg_loso import load_loso_split
from tsml_eval.experiments.experiments import run_classification_experiment


def run_native_loso_component(
    *,
    data_path: Path,
    results_path: Path,
    component: str,
    held_subject: int,
    dataset: str,
) -> None:
    result_dataset = f"{dataset}LOSO"
    result_name = f"GEAR-Comp-Native-{component}"
    prediction_dir = results_path / result_name / "Predictions" / result_dataset
    train_file = prediction_dir / f"trainResample{held_subject}.csv"
    test_file = prediction_dir / f"testResample{held_subject}.csv"
    if all(path.is_file() and path.stat().st_size > 0 for path in (train_file, test_file)):
        print(
            f"{result_name}/{result_dataset}/subject{held_subject}: "
            "complete native results exist; skipping."
        )
        return

    X_train, y_train, X_test, y_test, subjects = load_loso_split(
        data_path, dataset, held_subject
    )
    print(
        f"{result_name}/{result_dataset}/subject{held_subject}: "
        f"native {NATIVE_METHODS[component]}; subjects={len(subjects)}, "
        f"TRAIN {X_train.shape}, TEST {X_test.shape}"
    )
    classifier = GEARNativeComponentPipeline(
        component=component,
        random_state=held_subject,
        n_jobs=1,
    )
    run_classification_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier,
        results_path=str(results_path),
        classifier_name=result_name,
        dataset_name=result_dataset,
        resample_id=held_subject,
        build_train_file=True,
        build_test_file=True,
        benchmark_time=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("results_path", type=Path)
    parser.add_argument("component", choices=COMPONENTS)
    parser.add_argument("held_subject", type=int)
    parser.add_argument("--dataset", default="OpenCloseFist")
    args = parser.parse_args()
    run_native_loso_component(
        data_path=args.data_path,
        results_path=args.results_path,
        component=args.component,
        held_subject=args.held_subject,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
