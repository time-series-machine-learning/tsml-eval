# -*- coding: utf-8 -*-
"""Tests for results writing functions."""

__author__ = ["MatthewMiddlehurst"]

import os

import numpy as np
import pytest

from tsml_eval.utils.experiments import (
    _check_classification_third_line,
    _check_clustering_third_line,
    _check_first_line,
    _check_regression_third_line,
    _check_results_line,
    _check_second_line,
    fix_broken_second_line,
    validate_results_file,
    write_classification_results,
    write_clustering_results,
    write_regression_results,
)


def test_write_classification_results():
    """Test writing of classification results files."""
    class_labels, predictions, probabilities = _generate_labels_and_predictions()

    output_path = (
        "./test_output/classification/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/classification/"
    )

    write_classification_results(
        predictions,
        probabilities,
        class_labels,
        "Test",
        "Test",
        output_path,
        full_path=False,
        first_line_comment="test_write_classification_results",
    )

    _check_classification_file_format(
        f"{output_path}/Test/Predictions/Test/results.csv"
    )

    os.remove(f"{output_path}/Test/Predictions/Test/results.csv")


def _check_classification_file_format(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    assert _check_first_line(lines[0])
    assert _check_second_line(lines[1])
    assert _check_classification_third_line(lines[2])

    for i in range(3, 6):
        assert _check_results_line(lines[i])


def test_write_regression_results():
    """Test writing of regression results files."""
    labels, predictions, _ = _generate_labels_and_predictions()

    output_path = (
        "./test_output/regression/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/regression/"
    )

    write_regression_results(
        predictions,
        labels,
        "Test",
        "Test",
        output_path,
        full_path=False,
        first_line_comment="test_write_regression_results",
    )

    _check_regression_file_format(f"{output_path}/Test/Predictions/Test/results.csv")

    os.remove(f"{output_path}/Test/Predictions/Test/results.csv")


def _check_regression_file_format(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    assert _check_first_line(lines[0])
    assert _check_second_line(lines[1])
    assert _check_regression_third_line(lines[2])

    for i in range(3, 6):
        assert _check_results_line(lines[i], probabilities=False)


def test_write_clustering_results():
    """Test writing of clustering results files."""
    (
        class_labels,
        cluster_predictions,
        cluster_probabilities,
    ) = _generate_labels_and_predictions()

    output_path = (
        "./test_output/clustering/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/clustering/"
    )

    write_clustering_results(
        cluster_predictions,
        cluster_probabilities,
        class_labels,
        "Test",
        "Test",
        output_path,
        full_path=False,
        first_line_comment="test_write_clustering_results",
    )

    _check_clustering_file_format(f"{output_path}/Test/Predictions/Test/results.csv")

    os.remove(f"{output_path}/Test/Predictions/Test/results.csv")


def _check_clustering_file_format(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    assert _check_first_line(lines[0])
    assert _check_second_line(lines[1])
    assert _check_clustering_third_line(lines[2])

    for i in range(3, 6):
        assert _check_results_line(lines[i])


def _generate_labels_and_predictions():
    labels = np.random.randint(0, 2, 10)
    predictions = np.random.randint(0, 2, 10)

    probabilities = np.zeros((10, 3))
    for i in range(10):
        probabilities[i, predictions[i]] = 1

    return labels, predictions, probabilities


@pytest.mark.parametrize(
    "path",
    [
        "test_files/regressionResultsFile.csv",
        "test_files/classificationResultsFile1.csv",
    ],
)
def test_validate_results_file(path):
    assert validate_results_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "test_files/brokenRegressionResultsFile.csv",
        "test_files/brokenClassificationResultsFile.csv",
    ],
)
def test_validate_broken_results_file(path):
    assert not validate_results_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "test_files/regressionResultsFile.csv",
        "test_files/brokenRegressionResultsFile.csv",
    ],
)
def test_fix_broken_second_line(path):
    """Test that the second line of a broken results file is fixed."""
    output_path = (
        "./test_output/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/"
    )

    fix_broken_second_line(path, f"{output_path}/secondLineTest.csv")

    os.remove(f"{output_path}/secondLineTest.csv")
