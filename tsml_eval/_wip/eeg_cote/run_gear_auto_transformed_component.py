"""Run one HC2 component on an already saved GEAR-Auto transform.

This worker is intended for recovery of a GEAR-Auto HC2 result from its four
components. It does not fit GEAR again. Instead, it loads the transformed
TRAIN/TEST files and asks the component for the native train probabilities
used by HC2 (rather than running an external cross-validation).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from aeon.classification.convolution_based import Arsenal
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.interval_based import DrCIFClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier

from tsml_eval.experiments.experiments import run_classification_experiment
from tsml_eval.utils.datasets import load_experiment_data


COMPONENTS = ("Arsenal", "DrCIF", "STC", "TDE")


def _make_component(component: str, random_state: int):
    """Construct a component with the full default HC2 budget."""
    common = {"random_state": random_state, "n_jobs": 1}
    if component == "Arsenal":
        return Arsenal(n_kernels=2000, n_estimators=25, **common)
    if component == "DrCIF":
        return DrCIFClassifier(n_estimators=500, **common)
    if component == "STC":
        return ShapeletTransformClassifier(n_shapelet_samples=10000, **common)
    if component == "TDE":
        return TemporalDictionaryEnsemble(
            n_parameter_samples=250,
            max_ensemble_size=50,
            randomly_selected_params=50,
            **common,
        )
    raise ValueError(f"component must be one of {COMPONENTS}; found {component!r}.")


def run_component(
    transformed_data_root: Path,
    results_root: Path,
    component: str,
    dataset: str,
    resample_id: int,
) -> None:
    """Fit one component and write its native TRAIN and TEST results."""
    result_name = f"GEAR-Auto-Native-{component}"
    prediction_dir = results_root / result_name / "Predictions" / dataset
    train_result = prediction_dir / f"trainResample{resample_id}.csv"
    test_result = prediction_dir / f"testResample{resample_id}.csv"
    result_files = (train_result, test_result)
    if all(path.is_file() and path.stat().st_size > 0 for path in result_files):
        print(f"{result_name}/{dataset}: complete results exist; skipping.")
        return

    X_train, y_train, X_test, y_test, resample = load_experiment_data(
        str(transformed_data_root),
        dataset,
        resample_id,
        predefined_resample=False,
    )
    if resample:
        raise RuntimeError(
            "This recovery worker only supports resample 0 of the saved transform."
        )

    classifier = _make_component(component, random_state=resample_id)
    if not classifier.get_tag("capability:train_estimate", False, False):
        raise RuntimeError(
            f"{type(classifier).__name__} does not advertise native train estimates."
        )

    print(
        f"{result_name}/{dataset}: TRAIN {X_train.shape}, TEST {X_test.shape}; "
        "using the component's native HC2 train-estimate mechanism."
    )
    run_classification_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier,
        results_path=str(results_root),
        classifier_name=result_name,
        dataset_name=dataset,
        resample_id=resample_id,
        build_train_file=True,
        build_test_file=True,
        benchmark_time=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transformed_data_root", type=Path)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("component", choices=COMPONENTS)
    parser.add_argument("dataset")
    parser.add_argument("--resample-id", type=int, default=0)
    args = parser.parse_args()
    run_component(
        args.transformed_data_root,
        args.results_root,
        args.component,
        args.dataset,
        args.resample_id,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
