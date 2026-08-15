"""Run one GEAR component using HC2's native train-estimate mechanism.

The existing ``GEAR-Comp-*`` train files were produced by external ten-fold CV
of the complete transform/classifier pipeline. This recovery worker deliberately
writes to a separate ``GEAR-Comp-Native-*`` family. It fits the component-specific
GEAR reducer once, calls the HC2 component's public ``fit_predict_proba`` method,
and then predicts the test split from that same fitted component.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.utils import check_random_state

from tsml_eval.experiments._channel_selection_hc2 import (
    _make_gear_transformer,
    _metadata_to_builtin,
    _selector_metadata,
)
from tsml_eval.experiments._get_classifier import _make_hc2_or_component
from tsml_eval.experiments.experiments import run_classification_experiment
from tsml_eval.utils.datasets import load_experiment_data
from tsml_eval.utils.resampling import stratified_resample_data


COMPONENTS = ("Arsenal", "DrCIF", "STC", "TDE")
NATIVE_METHODS = {
    "Arsenal": "out-of-bag ensemble estimates",
    "DrCIF": "out-of-bag forest estimates",
    "STC": "RotationForest out-of-bag estimates",
    "TDE": "internal leave-one-out estimates",
}


def _is_native_result(path: Path, result_name: str, split: str) -> bool:
    """Return whether a result was produced by the native GEAR pipeline."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open(encoding="utf-8") as result_file:
        first_line = result_file.readline()
    expected = f",{result_name} (GEARNativeComponentPipeline),{split},"
    return expected in first_line


class GEARNativeComponentPipeline(BaseClassifier):
    """One component-specific GEAR view with the component's native estimate."""

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:train_estimate": True,
    }

    def __init__(self, component, random_state=None, n_jobs=1):
        self.component = component
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__()

    def _prepare_fit(self, X, y):
        component_key = self.component.casefold()
        if self.component not in COMPONENTS:
            raise ValueError(f"component must be one of {COMPONENTS}.")

        self.train_input_shape_ = tuple(int(value) for value in X.shape)
        self.reducer_ = _make_gear_transformer(
            component=component_key,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        start = time.perf_counter_ns()
        Xt, yt = self.reducer_.fit_resample(X, y)
        self.transform_fit_time_millis_ = (
            time.perf_counter_ns() - start
        ) / 1_000_000
        if len(yt) != len(y) or not np.array_equal(yt, y):
            raise RuntimeError(
                "GEAR component reducers must retain all training labels in order."
            )
        self.train_output_shape_ = tuple(int(value) for value in Xt.shape)
        self.component_ = _make_hc2_or_component(
            component=component_key,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            fit_contract=0,
            kwargs={},
        )
        return Xt, yt

    def _fit(self, X, y):
        Xt, yt = self._prepare_fit(X, y)
        start = time.perf_counter_ns()
        self.component_.fit(Xt, yt)
        self.component_fit_time_millis_ = (
            time.perf_counter_ns() - start
        ) / 1_000_000
        self.fit_time_millis_ = (
            self.transform_fit_time_millis_ + self.component_fit_time_millis_
        )
        return self

    def _fit_predict_proba(self, X, y):
        Xt, yt = self._prepare_fit(X, y)
        start = time.perf_counter_ns()
        probabilities = self.component_.fit_predict_proba(Xt, yt)
        self.component_fit_time_millis_ = (
            time.perf_counter_ns() - start
        ) / 1_000_000
        self.fit_time_millis_ = (
            self.transform_fit_time_millis_ + self.component_fit_time_millis_
        )
        return probabilities

    def _predict_proba(self, X):
        self.test_input_shape_ = tuple(int(value) for value in X.shape)
        start = time.perf_counter_ns()
        Xt = self.reducer_.transform(X)
        self.transform_predict_time_millis_ = (
            time.perf_counter_ns() - start
        ) / 1_000_000
        self.test_output_shape_ = tuple(int(value) for value in Xt.shape)

        start = time.perf_counter_ns()
        probabilities = self.component_.predict_proba(Xt)
        self.component_predict_time_millis_ = (
            time.perf_counter_ns() - start
        ) / 1_000_000
        return probabilities

    def _predict(self, X):
        probabilities = self._predict_proba(X)
        rng = check_random_state(self.random_state)
        return np.asarray(
            [
                self.classes_[rng.choice(np.flatnonzero(row == row.max()))]
                for row in probabilities
            ]
        )

    def get_experiment_metadata(self):
        if not hasattr(self, "reducer_"):
            return {}
        metadata = {
            "gear_mode": "component",
            "component": self.component,
            "train_estimate_method": NATIVE_METHODS[self.component],
            "timings_ms": {
                "transform_fit": getattr(
                    self, "transform_fit_time_millis_", None
                ),
                "component_fit_and_native_estimate": getattr(
                    self, "component_fit_time_millis_", None
                ),
                "transform_predict": getattr(
                    self, "transform_predict_time_millis_", None
                ),
                "component_predict": getattr(
                    self, "component_predict_time_millis_", None
                ),
            },
            "train_input_shape": getattr(self, "train_input_shape_", None),
            "train_output_shape": getattr(self, "train_output_shape_", None),
            "test_input_shape": getattr(self, "test_input_shape_", None),
            "test_output_shape": getattr(self, "test_output_shape_", None),
            "selector": _selector_metadata(getattr(self, "reducer_", None)),
        }
        if hasattr(self.reducer_, "get_reduction_summary"):
            summary = self.reducer_.get_reduction_summary()
            metadata["reduction_summary"] = {
                key: value
                for key, value in summary.items()
                if key not in {"case_indices", "time_indices"}
            }
        candidates = getattr(self.reducer_, "candidate_results_", None)
        if candidates is not None:
            records = candidates.to_dict("records")
            for record in records:
                if "fit_time" in record:
                    record["fit_time_seconds"] = record.pop("fit_time")
                if "predict_time" in record:
                    record["predict_time_seconds"] = record.pop("predict_time")
            metadata["reduction_candidates"] = records
        return _metadata_to_builtin(metadata)


def _load_data(data_path: Path, dataset: str, resample_id: int):
    X_train, y_train, X_test, y_test, resample = load_experiment_data(
        str(data_path),
        dataset,
        resample_id,
        predefined_resample=False,
    )
    if resample:
        X_train, y_train, X_test, y_test = stratified_resample_data(
            X_train,
            y_train,
            X_test,
            y_test,
            random_state=resample_id,
        )
    return X_train, y_train, X_test, y_test


def run_native_component(
    *,
    data_path: Path,
    results_path: Path,
    component: str,
    dataset: str,
    resample_id: int,
) -> None:
    result_name = f"GEAR-Comp-Native-{component}"
    prediction_dir = results_path / result_name / "Predictions" / dataset
    train_file = prediction_dir / f"trainResample{resample_id}.csv"
    test_file = prediction_dir / f"testResample{resample_id}.csv"
    if _is_native_result(train_file, result_name, "TRAIN") and _is_native_result(
        test_file, result_name, "TEST"
    ):
        print(f"{result_name}/{dataset}: complete native results exist; skipping.")
        return

    X_train, y_train, X_test, y_test = _load_data(
        data_path, dataset, resample_id
    )
    classifier = GEARNativeComponentPipeline(
        component=component,
        random_state=resample_id,
        n_jobs=1,
    )
    print(
        f"{result_name}/{dataset}: native {NATIVE_METHODS[component]}; "
        f"TRAIN {X_train.shape}, TEST {X_test.shape}"
    )
    run_classification_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier,
        results_path=str(results_path),
        classifier_name=result_name,
        dataset_name=dataset,
        resample_id=resample_id,
        build_train_file=True,
        build_test_file=True,
        benchmark_time=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("results_path", type=Path)
    parser.add_argument("component", choices=COMPONENTS)
    parser.add_argument("dataset")
    parser.add_argument("--resample-id", type=int, default=0)
    args = parser.parse_args()
    run_native_component(
        data_path=args.data_path,
        results_path=args.results_path,
        component=args.component,
        dataset=args.dataset,
        resample_id=args.resample_id,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
