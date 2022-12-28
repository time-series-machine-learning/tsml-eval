# -*- coding: utf-8 -*-

__author__ = ["MatthewMiddlehurst"]

import os

import numpy as np

from tsml_eval.utils.experiments import (
    write_classification_results,
    write_clustering_results,
    write_regression_results,
)


def test_write_classification_results():
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
    lines = open(file_path, "r").readlines()

    _check_first_line(lines[0])
    _check_second_line(lines[1])

    line = lines[2].split(",")
    float(line[0])
    float(line[1])
    float(line[2])
    float(line[3])
    float(line[4])
    float(line[5])
    float(line[7])
    float(line[8])

    for i in range(3, 6):
        _check_results_line(lines[i])


def test_write_regression_results():
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
    lines = open(file_path, "r").readlines()

    _check_first_line(lines[0])
    _check_second_line(lines[1])

    line = lines[2].split(",")
    float(line[0])
    float(line[1])
    float(line[2])
    float(line[3])
    float(line[4])
    float(line[6])
    float(line[7])

    for i in range(3, 6):
        _check_results_line(lines[i], probabilities=False)


def test_write_clustering_results():
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
    lines = open(file_path, "r").readlines()

    _check_first_line(lines[0])
    _check_second_line(lines[1])

    line = lines[2].split(",")
    float(line[0])
    float(line[1])
    float(line[2])
    float(line[3])
    float(line[4])
    float(line[5])
    float(line[6])

    for i in range(3, 6):
        _check_results_line(lines[i])


def _check_first_line(line):
    line = line.split(",")
    assert len(line) >= 5


def _check_second_line(line):
    line = line.split(",")
    assert len(line) >= 1


def _check_results_line(line, probabilities=True):
    line = line.split(",")
    assert len(line) >= 2

    float(line[0])
    float(line[1])

    if probabilities:
        assert len(line) >= 5

        assert line[2] == ""
        float(line[3])
        float(line[4])
    else:
        assert len(line) == 2


def _generate_labels_and_predictions():
    labels = np.random.randint(0, 2, 10)
    predictions = np.random.randint(0, 2, 10)

    probabilities = np.zeros((10, 3))
    for i in range(10):
        probabilities[i, predictions[i]] = 1

    return labels, predictions, probabilities
