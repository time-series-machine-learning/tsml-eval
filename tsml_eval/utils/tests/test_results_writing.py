"""Tests for results writing functions."""

__maintainer__ = ["MatthewMiddlehurst"]

import os

import numpy as np
import pytest

from tsml_eval.experiments.tests import (
    _CLASSIFIER_RESULTS_PATH,
    _CLUSTERER_RESULTS_PATH,
    _FORECASTER_RESULTS_PATH,
    _REGRESSOR_RESULTS_PATH,
)
from tsml_eval.utils.results_validation import (
    _check_classification_third_line,
    _check_clustering_third_line,
    _check_first_line,
    _check_forecasting_third_line,
    _check_regression_third_line,
    _check_results_lines,
    _check_second_line,
)
from tsml_eval.utils.results_writing import (
    write_classification_results,
    write_clustering_results,
    write_forecasting_results,
    write_regression_results,
    write_results_to_tsml_format,
)


def test_write_classification_results():
    """Test writing of classification results files."""
    class_labels, predictions, probabilities = _generate_labels_and_predictions()

    write_classification_results(
        predictions,
        probabilities,
        class_labels,
        "Test",
        "Test",
        _CLASSIFIER_RESULTS_PATH,
        full_path=False,
        first_line_comment="test_write_classification_results",
        n_classes=3,
    )

    _check_classification_file_format(
        f"{_CLASSIFIER_RESULTS_PATH}/Test/Predictions/Test/results.csv"
    )

    os.remove(f"{_CLASSIFIER_RESULTS_PATH}/Test/Predictions/Test/results.csv")


def _check_classification_file_format(file_path, num_results_lines=None):
    with open(file_path) as f:
        lines = f.readlines()

    assert _check_first_line(lines[0])
    assert _check_second_line(lines[1])
    assert _check_classification_third_line(lines[2])
    n_classes = int(lines[2].split(",")[5])

    _check_results_lines(lines, num_results_lines=num_results_lines, n_probas=n_classes)


def test_write_classification_results_invalid():
    """Test writing of classification results files with invalid input."""
    with pytest.raises(IndexError, match="The number of predicted values"):
        write_classification_results(
            np.zeros(10),
            np.zeros((11, 2)),
            np.zeros(12),
            "Test",
            "Test",
            "test_output",
        )

    with pytest.raises(IndexError, match="The number of classes is not"):
        write_classification_results(
            np.zeros(10),
            np.zeros((10, 3)),
            np.zeros(10),
            "Test",
            "Test",
            "test_output",
            n_classes=2,
        )


def test_write_regression_results():
    """Test writing of regression results files."""
    labels, predictions, _ = _generate_labels_and_predictions()

    write_regression_results(
        predictions,
        labels,
        "Test",
        "Test",
        _REGRESSOR_RESULTS_PATH,
        full_path=False,
        first_line_comment="test_write_regression_results",
    )

    _check_regression_file_format(
        f"{_REGRESSOR_RESULTS_PATH}/Test/Predictions/Test/results.csv"
    )

    os.remove(f"{_REGRESSOR_RESULTS_PATH}/Test/Predictions/Test/results.csv")


def _check_regression_file_format(file_path, num_results_lines=None):
    with open(file_path) as f:
        lines = f.readlines()

    assert _check_first_line(lines[0])
    assert _check_second_line(lines[1])
    assert _check_regression_third_line(lines[2])

    _check_results_lines(
        lines, num_results_lines=num_results_lines, probabilities=False
    )


def test_write_forecasting_results():
    """Test writing of forecasting results files."""
    labels, predictions, _ = _generate_labels_and_predictions()

    write_forecasting_results(
        predictions,
        labels,
        "Test",
        "Test",
        _FORECASTER_RESULTS_PATH,
        full_path=False,
        first_line_comment="test_write_forecasting_results",
    )

    _check_forecasting_file_format(
        f"{_FORECASTER_RESULTS_PATH}/Test/Predictions/Test/results.csv"
    )

    os.remove(f"{_FORECASTER_RESULTS_PATH}/Test/Predictions/Test/results.csv")


def _check_forecasting_file_format(file_path, num_results_lines=None):
    with open(file_path) as f:
        lines = f.readlines()

    assert _check_first_line(lines[0])
    assert _check_second_line(lines[1])
    assert _check_forecasting_third_line(lines[2])

    _check_results_lines(
        lines, num_results_lines=num_results_lines, probabilities=False
    )


def test_write_clustering_results():
    """Test writing of clustering results files."""
    (
        class_labels,
        cluster_predictions,
        cluster_probabilities,
    ) = _generate_labels_and_predictions()

    write_clustering_results(
        cluster_predictions,
        cluster_probabilities,
        class_labels,
        "Test",
        "Test",
        _CLUSTERER_RESULTS_PATH,
        full_path=False,
        first_line_comment="test_write_clustering_results",
        n_clusters=3,
    )

    _check_clustering_file_format(
        f"{_CLUSTERER_RESULTS_PATH}/Test/Predictions/Test/results.csv"
    )

    os.remove(f"{_CLUSTERER_RESULTS_PATH}/Test/Predictions/Test/results.csv")


def _check_clustering_file_format(file_path, num_results_lines=None):
    with open(file_path) as f:
        lines = f.readlines()

    assert _check_first_line(lines[0])
    assert _check_second_line(lines[1])
    assert _check_clustering_third_line(lines[2])
    n_probas = int(lines[2].split(",")[6])

    _check_results_lines(lines, num_results_lines=num_results_lines, n_probas=n_probas)


def test_write_clustering_results_invalid():
    """Test writing of clustering results files with invalid input."""
    with pytest.raises(IndexError, match="The number of predicted values"):
        write_clustering_results(
            np.zeros(10),
            np.zeros((11, 2)),
            np.zeros(12),
            "Test",
            "Test",
            "test_output",
        )

    with pytest.raises(IndexError, match="The number of clusters is not"):
        write_clustering_results(
            np.zeros(10),
            np.zeros((10, 3)),
            np.zeros(10),
            "Test",
            "Test",
            "test_output",
            n_clusters=2,
        )


def _generate_labels_and_predictions():
    labels = np.random.randint(0, 2, 10)
    predictions = np.random.randint(0, 2, 10)

    probabilities = np.zeros((10, 3))
    for i in range(10):
        probabilities[i, predictions[i]] = 1

    return labels, predictions, probabilities


def test_write_results_to_tsml_format_invalid():
    """Test writing of results files with invalid input."""
    with pytest.raises(IndexError, match="The number of predicted values"):
        write_results_to_tsml_format(
            np.zeros(10),
            np.zeros(11),
            "Test",
            "Test",
            "test_output",
        )

    with pytest.raises(ValueError, match="Unknown 'split' value"):
        write_results_to_tsml_format(
            np.zeros(10),
            np.zeros(10),
            "Test",
            "Test",
            "test_output",
            split="invalid",
        )
