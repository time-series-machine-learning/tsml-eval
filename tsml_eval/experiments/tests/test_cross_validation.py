"""Tests for cross-validation functions."""

import os

import numpy as np
import pytest
from aeon.classification import DummyClassifier
from aeon.datasets import load_covid_3month, load_unit_test
from aeon.regression import DummyRegressor
from aeon.transformations.collection import Normalizer

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


def test_cross_validation_clones_and_indexes_list_data(tmp_path):
    """Test fold state isolation, transforms, list indexing and attribute paths."""
    X, y = load_unit_test()
    X = list(X)
    classifier = DummyClassifier()
    results_path = str(tmp_path / "results")
    attribute_path = str(tmp_path / "attributes")

    classification_cross_validation(
        X,
        y,
        classifier,
        results_path,
        cv=2,
        fold_ids=0,
        classifier_name="DummyClassifierListCV",
        dataset_name="UnitTest",
        attribute_file_path=attribute_path,
        benchmark_time=False,
        data_transforms=Normalizer(),
        overwrite=True,
    )

    test_file = (
        tmp_path
        / "results"
        / "DummyClassifierListCV"
        / "Predictions"
        / "UnitTest"
        / "testResample0.csv"
    )
    attribute_file = tmp_path / "attributes" / "fold0" / "DummyClassifier.txt"
    assert test_file.exists()
    assert attribute_file.exists()
    assert not hasattr(classifier, "classes_")


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
