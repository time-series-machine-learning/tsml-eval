"""Tests for clustering experiments."""

__author__ = ["MatthewMiddlehurst"]

import os
import runpy

import pytest
from tsml.dummy import DummyClassifier, DummyClusterer

from tsml_eval.experiments import (
    clustering_experiments,
    run_clustering_experiment,
    set_clusterer,
    threaded_clustering_experiments,
)
from tsml_eval.experiments.tests import _CLUSTERER_RESULTS_PATH
from tsml_eval.utils.test_utils import (
    _TEST_DATA_PATH,
    _check_set_method,
    _check_set_method_results,
)
from tsml_eval.utils.tests.test_results_writing import _check_clustering_file_format


@pytest.mark.parametrize(
    "clusterer",
    ["DummyClusterer-tsml", "DummyClusterer-aeon", "DummyClusterer-sklearn"],
)
@pytest.mark.parametrize(
    "dataset",
    ["MinimalChinatown", "UnequalMinimalChinatown", "EqualMinimalJapaneseVowels"],
)
def test_run_clustering_experiment(clusterer, dataset):
    """Test clustering experiments with test data and clusterer."""
    if clusterer == "DummyClusterer-aeon" and dataset == "UnequalMinimalChinatown":
        return  # todo remove when aeon kmeans supports unequal

    args = [
        _TEST_DATA_PATH,
        _CLUSTERER_RESULTS_PATH,
        clusterer,
        dataset,
        "0",
        "-te",
    ]

    clustering_experiments.run_experiment(args)

    test_file = (
        f"{_CLUSTERER_RESULTS_PATH}{clusterer}/Predictions/{dataset}/testResample0.csv"
    )
    train_file = (
        f"{_CLUSTERER_RESULTS_PATH}{clusterer}/Predictions/{dataset}/trainResample0.csv"
    )

    assert os.path.exists(test_file) and os.path.exists(train_file)

    _check_clustering_file_format(test_file)
    _check_clustering_file_format(train_file)

    # test present results checking
    clustering_experiments.run_experiment(args)

    os.remove(test_file)
    os.remove(train_file)


def test_run_clustering_experiment_main():
    """Test clustering experiments main with test data and clusterer."""
    clusterer = "KMeans"
    dataset = "MinimalChinatown"

    # run twice to test results present check
    for _ in range(2):
        runpy.run_path(
            "./tsml_eval/experiments/clustering_experiments.py"
            if os.getcwd().split("\\")[-1] != "tests"
            else "../clustering_experiments.py",
            run_name="__main__",
        )

    train_file = (
        f"{_CLUSTERER_RESULTS_PATH}{clusterer}/Predictions/{dataset}/trainResample0.csv"
    )
    assert os.path.exists(train_file)
    _check_clustering_file_format(train_file)

    os.remove(
        f"{_CLUSTERER_RESULTS_PATH}{clusterer}/Predictions/{dataset}/trainResample0.csv"
    )


def test_run_threaded_clustering_experiment():
    """Test threaded clustering experiments with test data and clusterer."""
    clusterer = "KMeans"
    dataset = "MinimalChinatown"

    args = [
        _TEST_DATA_PATH,
        _CLUSTERER_RESULTS_PATH,
        clusterer,
        dataset,
        "1",
        "-nj",
        "2",
        # also test normalisation here
        "--row_normalise",
        "-te",
    ]

    threaded_clustering_experiments.run_experiment(args)

    train_file = (
        f"{_CLUSTERER_RESULTS_PATH}{clusterer}/Predictions/{dataset}/trainResample1.csv"
    )
    assert os.path.exists(train_file)
    _check_clustering_file_format(train_file)

    # test present results checking
    threaded_clustering_experiments.run_experiment(args)

    # this covers the main method and experiment function result file checking
    runpy.run_path(
        "./tsml_eval/experiments/threaded_clustering_experiments.py"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../threaded_clustering_experiments.py",
        run_name="__main__",
    )

    os.remove(train_file)


def test_run_clustering_experiment_invalid_build_settings():
    """Test run_clustering_experiment method with invalid build settings."""
    with pytest.raises(ValueError, match="Both test_file and train_file"):
        run_clustering_experiment(
            [],
            [],
            None,
            "",
            build_train_file=False,
            build_test_file=False,
        )


def test_run_clustering_experiment_invalid_estimator():
    """Test run_clustering_experiment method with invalid estimator."""
    with pytest.raises(TypeError, match="clusterer must be a"):
        run_clustering_experiment(
            [],
            [],
            DummyClassifier(),
            "",
        )


def test_set_clusterer():
    """Test set_clusterer method."""
    clusterer_lists = [
        set_clusterer.distance_based_clusterers,
        set_clusterer.other_clusterers,
        set_clusterer.vector_clusterers,
    ]

    clusterer_dict = {}
    all_clusterer_names = []

    for clusterer_list in clusterer_lists:
        _check_set_method(
            set_clusterer.set_clusterer,
            clusterer_list,
            clusterer_dict,
            all_clusterer_names,
        )

    _check_set_method_results(
        clusterer_dict, estimator_name="Clusterers", method_name="set_clusterer"
    )


def test_set_clusterer_invalid():
    """Test set_clusterer method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN CLUSTERER"):
        set_clusterer.set_clusterer("invalid")


@pytest.mark.parametrize("n_clusters", ["4", "-1"])
@pytest.mark.parametrize(
    "clusterer",
    ["DummyClusterer-tsml", "DummyClusterer-aeon", "DummyClusterer-sklearn"],
)
def test_n_clusters(n_clusters, clusterer):
    """Test n_clusters parameter."""
    dataset = "MinimalChinatown"

    args = [
        _TEST_DATA_PATH,
        _CLUSTERER_RESULTS_PATH,
        clusterer,
        dataset,
        "1",
        "--n_clusters",
        n_clusters,
        "-ow",
    ]

    clustering_experiments.run_experiment(args)

    train_file = (
        f"{_CLUSTERER_RESULTS_PATH}{clusterer}/Predictions/{dataset}/trainResample1.csv"
    )

    assert os.path.exists(train_file)

    _check_clustering_file_n_clusters(
        train_file, "2" if n_clusters == "-1" else n_clusters
    )

    os.remove(train_file)


def _check_clustering_file_n_clusters(file_path, expected):
    with open(file_path, "r") as f:
        lines = f.readlines()

    line = lines[2].split(",")
    assert line[6].strip() == expected


def test_invalid_n_clusters():
    with pytest.raises(ValueError, match="n_clusters must be a"):
        run_clustering_experiment(
            [],
            [],
            DummyClusterer(),
            "",
            n_clusters="invalid",
        )
