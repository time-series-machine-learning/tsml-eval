# -*- coding: utf-8 -*-
"""Tests for clustering experiments."""

__author__ = ["MatthewMiddlehurst"]

import os

from tsml_eval.experiments.clustering_experiments import run_experiment
from tsml_eval.utils.tests.test_results_writing import _check_clustering_file_format


def test_run_clustering_experiment():
    """Test clustering experiments with test data and clusterer."""
    result_path = (
        "./test_output/clustering/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/clustering/"
    )
    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../datasets/"
    )
    clusterer = "KMeans"
    dataset = "UnitTest"

    args = [
        None,
        data_path,
        result_path,
        clusterer,
        dataset,
        "1",
    ]
    run_experiment(args, overwrite=True)

    test_file = f"{result_path}{clusterer}/Predictions/{dataset}/testResample0.csv"
    train_file = f"{result_path}{clusterer}/Predictions/{dataset}/trainResample0.csv"

    assert os.path.exists(test_file) and os.path.exists(train_file)

    _check_clustering_file_format(test_file)
    _check_clustering_file_format(train_file)

    os.remove(test_file)
    os.remove(train_file)
