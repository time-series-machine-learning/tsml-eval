"""Tests for cross-validation functions."""

import os

import numpy as np
from aeon.classification import DummyClassifier
from aeon.datasets import load_covid_3month, load_unit_test
from aeon.regression import DummyRegressor

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
