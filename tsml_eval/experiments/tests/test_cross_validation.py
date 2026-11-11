"""Tests for cross-validation functions."""

import os

import numpy as np
import pytest
from aeon.classification import DummyClassifier
from aeon.datasets import load_covid_3month, load_unit_test
from aeon.regression import DummyRegressor
from aeon.transformations.collection import Normalizer
from sklearn.base import BaseEstimator

from tsml_eval.experiments import cross_validation as cv_module
from tsml_eval.experiments.cross_validation import (
    classification_cross_validation,
    classification_cross_validation_folds,
    regression_cross_validation,
    regression_cross_validation_folds,
)
from tsml_eval.experiments.tests import (
    _CLASSIFIER_RESULTS_PATH,
    _REGRESSOR_RESULTS_PATH,
)
from tsml_eval.utils.tests.test_results_writing import (
    _check_classification_file_format,
    _check_regression_file_format,
)


def test_classification_cross_validation():
    """Test the classification cross-validation function."""
    X, y = load_unit_test()
    classification_cross_validation(
        X,
        y,
        DummyClassifier(),
        _CLASSIFIER_RESULTS_PATH,
        classifier_name="DummyClassifierCV",
        benchmark_time=False,
        overwrite=True,
    )

    for i in range(10):
        test_file = (
            f"{_CLASSIFIER_RESULTS_PATH}/DummyClassifierCV/Predictions/"
            f"testResample{i}.csv"
        )

        assert os.path.exists(test_file)
        _check_classification_file_format(test_file)

    folds = classification_cross_validation_folds(X, y)
    assert len(folds) == 10
    assert len(folds[0]) == 2
    assert isinstance(folds[0][0], np.ndarray)
    assert isinstance(folds[0][1], np.ndarray)


def test_regression_cross_validation():
    """Test the classification cross-validation function."""
    X, y = load_covid_3month()
    regression_cross_validation(
        X,
        y,
        DummyRegressor(),
        _REGRESSOR_RESULTS_PATH,
        regressor_name="DummyRegressorCV",
        benchmark_time=False,
        data_transforms=Normalizer(),
        overwrite=True,
    )

    for i in range(10):
        test_file = (
            f"{_REGRESSOR_RESULTS_PATH}/DummyRegressorCV/Predictions/"
            f"testResample{i}.csv"
        )

        assert os.path.exists(test_file)
        _check_regression_file_format(test_file)

    folds = regression_cross_validation_folds(X, y)
    assert len(folds) == 10
    assert len(folds[0]) == 2
    assert isinstance(folds[0][0], np.ndarray)
    assert isinstance(folds[0][1], np.ndarray)


def test_classification_cross_validation_reduces_default_folds():
    """Test that both classification CV APIs reduce folds for small classes."""
    X = np.zeros((12, 1, 3))
    y = np.array([0] * 6 + [1] * 6)

    folds = classification_cross_validation_folds(X, y)

    assert len(folds) == 6


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("transform_list", [False, True])
def test_cross_validation_clones_and_indexes_list_data(
    tmp_path, monkeypatch, task, transform_list
):
    """Check fitted state and training membership across genuinely unequal folds."""
    X = [np.full((1, 3 + i % 3), float(i)) for i in range(12)]
    y = np.tile([0, 1], 6) if task == "classification" else np.arange(12, dtype=float)
    estimator = DummyClassifier() if task == "classification" else DummyRegressor()
    transform = _RecordingTransform()
    transforms = [transform] if transform_list else transform
    run_cv = getattr(cv_module, f"{task}_cross_validation")
    folds = getattr(cv_module, f"{task}_cross_validation_folds")(X, y, cv=2)
    runner_name = f"run_{task}_experiment"
    runner = getattr(cv_module, runner_name)
    fitted = []

    def record_fold(*args, **kwargs):
        fold_transform = kwargs["data_transforms"]
        if isinstance(fold_transform, list):
            fold_transform = fold_transform[0]
        fitted.append((args[4], fold_transform))
        return runner(*args, **kwargs)

    monkeypatch.setattr(cv_module, runner_name, record_fold)
    run_cv(
        X,
        y,
        estimator,
        str(tmp_path / "results"),
        cv=2,
        dataset_name="Unequal",
        attribute_file_path=str(tmp_path / "attributes"),
        benchmark_time=False,
        data_transforms=transforms,
    )

    assert len(fitted) == 2
    assert fitted[0][0] is not fitted[1][0]
    assert fitted[0][1] is not fitted[1][1]
    assert not estimator.is_fitted
    assert not hasattr(transform, "fit_ids_")
    if task == "classification":
        assert len(estimator.classes_) == 0

    name = type(estimator).__name__
    check_format = (
        _check_classification_file_format
        if task == "classification"
        else _check_regression_file_format
    )
    for fold, (train, test) in enumerate(folds):
        fold_estimator, fold_transform = fitted[fold]
        assert fold_estimator is not estimator
        assert fold_estimator.is_fitted
        assert fold_transform is not transform
        assert fold_transform.fit_ids_ == tuple(train)
        assert set(fold_transform.fit_ids_).isdisjoint(test)
        test_file = (
            tmp_path
            / "results"
            / name
            / "Predictions"
            / "Unequal"
            / f"testResample{fold}.csv"
        )
        check_format(str(test_file), num_results_lines=len(test))
        assert (tmp_path / "attributes" / f"fold{fold}" / f"{name}.txt").exists()


class _RecordingTransform(BaseEstimator):
    """Learn an offset and record training cases, rejecting fitted-state reuse."""

    def fit_transform(self, X, y=None):
        assert not hasattr(self, "fit_ids_")
        self.fit_ids_ = tuple(int(series[0, 0]) for series in X)
        self.offset_ = np.mean(self.fit_ids_)
        return self.transform(X)

    def transform(self, X, y=None):
        return [series - self.offset_ for series in X]


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("transform_list", [False, True])
@pytest.mark.parametrize("ignore_custom", [False, True])
def test_cross_validation_rejects_transformed_train_estimates(
    tmp_path, monkeypatch, task, transform_list, ignore_custom
):
    """Reject leaking train estimates before fitting or writing any result files."""

    def unexpected_run(*args, **kwargs):
        pytest.fail("Unsafe transform/train-estimate combination reached the runner")

    monkeypatch.setattr(cv_module, f"run_{task}_experiment", unexpected_run)
    transform = _RecordingTransform()
    extra = {"transform_train_only": True} if task == "classification" else {}
    with pytest.raises(ValueError, match="cannot be combined with data_transforms"):
        getattr(cv_module, f"{task}_cross_validation")(
            np.zeros((12, 1, 3)),
            np.tile([0, 1], 6),
            DummyClassifier() if task == "classification" else DummyRegressor(),
            str(tmp_path),
            cv=2,
            build_train_file=True,
            data_transforms=[transform] if transform_list else transform,
            ignore_custom_train_estimate=ignore_custom,
            benchmark_time=False,
            **extra,
        )
    assert not list(tmp_path.iterdir())
    assert not hasattr(transform, "fit_ids_")


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("transforms", [None, []])
def test_cross_validation_train_estimates_without_transforms(
    tmp_path, task, transforms
):
    """Keep train-only estimates available with no external preprocessing."""
    estimator = DummyClassifier() if task == "classification" else DummyRegressor()
    y = np.tile([0, 1], 6) if task == "classification" else np.arange(12) / 10
    getattr(cv_module, f"{task}_cross_validation")(
        np.zeros((12, 1, 3)),
        y,
        estimator,
        str(tmp_path),
        cv=2,
        fold_ids=0,
        build_test_file=False,
        build_train_file=True,
        data_transforms=transforms,
        benchmark_time=False,
    )
    directory = tmp_path / type(estimator).__name__ / "Predictions"
    check_format = (
        _check_classification_file_format
        if task == "classification"
        else _check_regression_file_format
    )
    check_format(str(directory / "trainResample0.csv"), num_results_lines=6)
    assert not (directory / "testResample0.csv").exists()


@pytest.mark.parametrize(
    ("fold_ids", "error_type"),
    [
        ([], ValueError),
        ([-1], ValueError),
        ([10], ValueError),
        ([0, 0], ValueError),
        (["0"], TypeError),
    ],
)
def test_cross_validation_invalid_fold_ids(fold_ids, error_type):
    """Test that invalid fold selections fail before running experiments."""
    X, y = load_unit_test()

    with pytest.raises(error_type):
        classification_cross_validation(
            X,
            y,
            DummyClassifier(),
            _CLASSIFIER_RESULTS_PATH,
            fold_ids=fold_ids,
            benchmark_time=False,
        )


def test_cross_validation_existing_result_policy(tmp_path):
    """Test that existing folds are skipped unless overwrite is requested."""
    X, y = load_unit_test()
    results_path = str(tmp_path)
    test_file = (
        tmp_path
        / "DummyClassifierSkipCV"
        / "Predictions"
        / "UnitTest"
        / "testResample0.csv"
    )
    test_file.parent.mkdir(parents=True)
    test_file.write_text("existing result", encoding="utf-8")

    classification_cross_validation(
        X,
        y,
        DummyClassifier(),
        results_path,
        cv=2,
        fold_ids=0,
        classifier_name="DummyClassifierSkipCV",
        dataset_name="UnitTest",
        benchmark_time=False,
    )
    assert test_file.read_text(encoding="utf-8") == "existing result"

    classification_cross_validation(
        X,
        y,
        DummyClassifier(),
        results_path,
        cv=2,
        fold_ids=0,
        classifier_name="DummyClassifierSkipCV",
        dataset_name="UnitTest",
        benchmark_time=False,
        overwrite=True,
    )
    _check_classification_file_format(str(test_file))
