"""Build one HC2 result from separately generated component result files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from tsml_eval.estimators.classification.hybrid import FromFileHIVECOTE
from tsml_eval.evaluation.storage import ClassifierResults
from tsml_eval.experiments.experiments import run_classification_experiment

_COMPONENTS = (
    ("ShapeletBased", "STC"),
    ("IntervalBased", "DrCIF-500"),
    ("ConvolutionBased", "Arsenal"),
    ("DictionaryBased", "TDE"),
)


def _component_directory(
    results_root: Path, category: str, classifier: str, dataset: str
) -> Path:
    return results_root / category / classifier / "Predictions" / dataset


def build_hc2(results_root: Path, dataset: str, resample: int) -> None:
    """Combine four complete HC2 component result pairs into an HC2 test result."""
    final_result = (
        results_root
        / "Hybrid"
        / "HC2"
        / "Predictions"
        / dataset
        / f"testResample{resample}.csv"
    )
    if final_result.is_file() and final_result.stat().st_size > 0:
        print(f"HC2 result already present; skipping: {final_result}")
        return

    component_directories = [
        _component_directory(results_root, category, classifier, dataset)
        for category, classifier in _COMPONENTS
    ]
    missing = [
        path / f"{split}Resample{resample}.csv"
        for path in component_directories
        for split in ("train", "test")
        if not (path / f"{split}Resample{resample}.csv").is_file()
    ]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing HC2 component results:\n{formatted}")

    classifier = FromFileHIVECOTE(
        classifiers=[f"{path}{os.sep}" for path in component_directories],
        alpha=4,
        random_state=resample,
    )
    first_component = component_directories[0]
    train_results = ClassifierResults().load_from_file(
        first_component / f"trainResample{resample}.csv"
    )
    test_results = ClassifierResults().load_from_file(
        first_component / f"testResample{resample}.csv"
    )
    # FromFileHIVECOTE uses the component files for all model information. Only
    # the number of cases and labels are needed from X/y, so avoid loading a
    # potentially very large original dataset during this lightweight build.
    X_train = np.zeros((train_results.n_cases, 1, 1), dtype=np.float32)
    X_test = np.zeros((test_results.n_cases, 1, 1), dtype=np.float32)
    run_classification_experiment(
        X_train,
        np.asarray(train_results.class_labels),
        X_test,
        np.asarray(test_results.class_labels),
        classifier,
        str(results_root / "Hybrid"),
        classifier_name="HC2",
        dataset_name=dataset,
        resample_id=resample,
        build_train_file=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("dataset")
    parser.add_argument("resample", type=int, nargs="?", default=0)
    args = parser.parse_args()
    build_hc2(args.results_root, args.dataset, args.resample)


if __name__ == "__main__":
    main()
