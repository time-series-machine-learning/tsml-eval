"""Tests for the results IO functionality."""

import os

import pytest

from tsml_eval.evaluation.storage.classifier_results import ClassifierResults
from tsml_eval.evaluation.storage.clusterer_results import ClustererResults
from tsml_eval.evaluation.storage.forecaster_results import ForecasterResults
from tsml_eval.evaluation.storage.regressor_results import RegressorResults
from tsml_eval.testing.testing_utils import _TEST_OUTPUT_PATH, _TEST_RESULTS_PATH
from tsml_eval.utils.results_validation import validate_results_file


def test_classifier_results():
    """Test ClassifierResults loading and saving."""
    cr = ClassifierResults().load_from_file(
        _TEST_RESULTS_PATH
        + "/classification/ROCKET/Predictions/MinimalChinatown/testResample0.csv"
    )
    cr.save_to_file(_TEST_OUTPUT_PATH + "/classification/results_io/")

    assert validate_results_file(
        _TEST_OUTPUT_PATH + "/classification/results_io/testResample0.csv"
    )

    os.remove(_TEST_OUTPUT_PATH + "/classification/results_io/testResample0.csv")


def test_java_classifier_results():
    """Test ClassifierResults loading and saving."""
    cr = ClassifierResults().load_from_file(
        _TEST_RESULTS_PATH + "/classification/javaResultsFile.csv"
    )
    cr.save_to_file(_TEST_OUTPUT_PATH + "/classification/results_io/java/")

    assert validate_results_file(
        _TEST_OUTPUT_PATH + "/classification/results_io/java/testResample0.csv"
    )

    os.remove(_TEST_OUTPUT_PATH + "/classification/results_io/java/testResample0.csv")


def test_clusterer_results():
    """Test ClustererResults loading and saving."""
    cr = ClustererResults().load_from_file(
        _TEST_RESULTS_PATH
        + "/clustering/KMeans/Predictions/MinimalChinatown/trainResample0.csv"
    )
    cr.save_to_file(_TEST_OUTPUT_PATH + "/clustering/results_io/")

    assert validate_results_file(
        _TEST_OUTPUT_PATH + "/clustering/results_io/trainResample0.csv"
    )

    os.remove(_TEST_OUTPUT_PATH + "/clustering/results_io/trainResample0.csv")


def test_regressor_results():
    """Test RegressorResults loading and saving."""
    cr = RegressorResults().load_from_file(
        _TEST_RESULTS_PATH
        + "/regression/ROCKET/Predictions/MinimalGasPrices/testResample0.csv"
    )
    cr.save_to_file(_TEST_OUTPUT_PATH + "/regression/results_io/")

    assert validate_results_file(
        _TEST_OUTPUT_PATH + "/regression/results_io/testResample0.csv"
    )

    os.remove(_TEST_OUTPUT_PATH + "/regression/results_io/testResample0.csv")


def test_forecaster_results():
    """Test ForecasterResults loading and saving."""
    cr = ForecasterResults().load_from_file(
        _TEST_RESULTS_PATH
        + "/forecasting/NaiveForecaster/Predictions/ShampooSales/testResample0.csv"
    )
    cr.save_to_file(_TEST_OUTPUT_PATH + "/forecasting/results_io/")

    assert validate_results_file(
        _TEST_OUTPUT_PATH + "/forecasting/results_io/testResample0.csv"
    )

    os.remove(_TEST_OUTPUT_PATH + "/forecasting/results_io/testResample0.csv")


results_classes = [
    ClassifierResults,
    ClustererResults,
    RegressorResults,
    ForecasterResults,
]


@pytest.mark.parametrize(
    "type,path",
    [
        (
            ClassifierResults,
            _TEST_RESULTS_PATH
            + "/classification/ROCKET/Predictions/MinimalChinatown/testResample0.csv",
        ),
        (
            ClustererResults,
            _TEST_RESULTS_PATH
            + "/clustering/KMeans/Predictions/MinimalChinatown/trainResample0.csv",
        ),
        (
            RegressorResults,
            _TEST_RESULTS_PATH
            + "/regression/ROCKET/Predictions/MinimalGasPrices/testResample0.csv",
        ),
        (
            ForecasterResults,
            _TEST_RESULTS_PATH
            + "/forecasting/NaiveForecaster/Predictions/ShampooSales/testResample0.csv",
        ),
    ],
)
def test_invalid_tasks_fail_to_load(type, path):
    """Test that loading into the wrong learning task class fails."""
    type().load_from_file(path)
    invalid_types = [x for x in results_classes if x != type]

    for invalid_type in invalid_types:
        with pytest.raises((ValueError, IndexError, AssertionError)):
            invalid_type().load_from_file(path)
