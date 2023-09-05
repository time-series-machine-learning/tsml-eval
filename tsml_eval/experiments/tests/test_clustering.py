"""Tests for clustering experiments."""

__author__ = ["MatthewMiddlehurst"]

import os

import pytest

from tsml_eval.experiments import set_clusterer
from tsml_eval.experiments.clustering_experiments import run_experiment
from tsml_eval.utils.test_utils import _check_set_method, _check_set_method_results
from tsml_eval.utils.tests.test_results_writing import _check_clustering_file_format


@pytest.mark.parametrize(
    "clusterer",
    ["DummyClusterer-tsml", "DummyClusterer-aeon", "DummyClusterer-sklearn"],
)
@pytest.mark.parametrize(
    "dataset",
    ["MinimalChinatown", "UnequalMinimalChinatown", "MinimalEqualJapaneseVowels"],
)
def test_run_clustering_experiment(clusterer, dataset):
    """Test clustering experiments with test data and clusterer."""
    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../datasets/"
    )
    result_path = (
        "./test_output/clustering/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/clustering/"
    )

    if clusterer == "DummyClusterer-aeon" and dataset == "UnequalMinimalChinatown":
        return  # todo remove when aeon kmeans supports unequal

    args = [
        data_path,
        result_path,
        clusterer,
        dataset,
        "0",
        "-te",
        "-ow",
    ]

    run_experiment(args)

    test_file = f"{result_path}{clusterer}/Predictions/{dataset}/testResample0.csv"
    train_file = f"{result_path}{clusterer}/Predictions/{dataset}/trainResample0.csv"

    assert os.path.exists(test_file) and os.path.exists(train_file)

    _check_clustering_file_format(test_file)
    _check_clustering_file_format(train_file)

    os.remove(test_file)
    os.remove(train_file)


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


@pytest.mark.parametrize("n_clusters", ["4", "-1"])
@pytest.mark.parametrize(
    "clusterer",
    ["DummyClusterer-aeon", "DummyClusterer-sklearn"],
)
def test_n_clusters(n_clusters, clusterer):
    """Test n_clusters parameter."""
    dataset = "MinimalChinatown"

    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../datasets/"
    )
    result_path = (
        f"./test_output/n_clusters/{n_clusters}/"
        if os.getcwd().split("\\")[-1] != "tests"
        else f"../../../test_output/n_clusters/{n_clusters}/"
    )

    args = [
        data_path,
        result_path,
        clusterer,
        dataset,
        "0",
        "--n_clusters",
        n_clusters,
        "-ow",
    ]

    run_experiment(args)

    train_file = f"{result_path}{clusterer}/Predictions/{dataset}/trainResample0.csv"

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
