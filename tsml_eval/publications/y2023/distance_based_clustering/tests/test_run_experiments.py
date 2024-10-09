"""Test file checking that the distance clustering experiments run correctly."""

import os
import runpy

from tsml_eval.publications.y2023.distance_based_clustering import _run_experiment
from tsml_eval.publications.y2023.distance_based_clustering.tests import (
    _DISTANCE_TEST_RESULTS_PATH,
)
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH
from tsml_eval.utils.tests.test_results_writing import _check_clustering_file_format


def test_run_distance_based_clustering_experiment():
    """Test paper classification experiments with test data and classifier."""
    classifier = "KMeans-dtw"
    dataset = "MinimalChinatown"
    resample = 0

    args = [
        _TEST_DATA_PATH,
        _DISTANCE_TEST_RESULTS_PATH,
        classifier,
        dataset,
        resample,
        "-ow",
    ]

    _run_experiment(args)

    train_file = (
        f"{_DISTANCE_TEST_RESULTS_PATH}{classifier}/Predictions/{dataset}/"
        "trainResample0.csv"
    )
    test_file = (
        f"{_DISTANCE_TEST_RESULTS_PATH}{classifier}/Predictions/{dataset}/"
        "testResample0.csv"
    )
    assert os.path.exists(train_file)
    assert os.path.exists(test_file)
    _check_clustering_file_format(train_file)
    _check_clustering_file_format(test_file)

    # this covers both the main method and present result file checking
    runpy.run_path(
        (
            "./tsml_eval/publications/y2023/distance_based_clustering/"
            "run_distance_experiments.py"
            if os.getcwd().split("\\")[-1] != "tests"
            else "../run_distance_experiments.py"
        ),
        run_name="__main__",
    )

    os.remove(train_file)
    os.remove(test_file)
