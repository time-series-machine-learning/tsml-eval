# -*- coding: utf-8 -*-
"""Tests for clustering experiments."""

__author__ = ["MatthewMiddlehurst"]

import os

import pytest

from tsml_eval.experiments import set_clusterer
from tsml_eval.experiments.clustering_experiments import run_experiment
from tsml_eval.utils.tests.test_results_writing import _check_clustering_file_format


@pytest.mark.parametrize(
    "clusterer",
    ["DummyClusterer-tsml", "DummyClusterer-aeon", "DummyClusterer-sklearn"],
)
@pytest.mark.parametrize(
    "dataset",
    ["MinimalChinatown", "UnequalMinimalChinatown", "MinimalJapaneseVowels"],
)
def test_run_clustering_experiment(clusterer, dataset):
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

    args = [
        None,
        data_path,
        result_path,
        clusterer,
        dataset,
        "0",
    ]

    # aeon estimators don't support unequal length series lists currently
    try:
        run_experiment(args, overwrite=True)
    except ValueError as e:
        if "not support unequal length series" in str(e):
            return
        else:
            raise e

    test_file = f"{result_path}{clusterer}/Predictions/{dataset}/testResample0.csv"
    train_file = f"{result_path}{clusterer}/Predictions/{dataset}/trainResample0.csv"

    assert os.path.exists(test_file) and os.path.exists(train_file)

    _check_clustering_file_format(test_file)
    _check_clustering_file_format(train_file)

    os.remove(test_file)
    os.remove(train_file)


def test_set_clusterer():
    clusterer_lists = [
        set_clusterer.distance_based_clusterers,
        set_clusterer.other_clusterers,
        set_clusterer.vector_clusterers,
    ]

    clusterer_dict = {}
    all_clusterer_names = []

    for clusterer_list in clusterer_lists:
        _check_set_clusterer(clusterer_list, clusterer_dict, all_clusterer_names)

    if not all(clusterer_dict.values()):
        missing_keys = [key for key, value in clusterer_dict.items() if not value]

        raise ValueError(
            "All clusterers seen in set_clusterer must have an entry for the full "
            "class name (usually with default parameters). Clusterers with missing "
            f"entries: {missing_keys}."
        )


def _check_set_clusterer(clusterer_sub_list, clusterer_dict, all_clusterer_names):
    for clusterer_names in clusterer_sub_list:
        clusterer_names = (
            [clusterer_names] if isinstance(clusterer_names, str) else clusterer_names
        )

        for clusterer_alias in clusterer_names:
            assert clusterer_alias not in all_clusterer_names
            all_clusterer_names.append(clusterer_alias)

            try:
                c = set_clusterer.set_clusterer(clusterer_alias)
            except ModuleNotFoundError:
                continue

            c_name = c.__class__.__name__
            if c_name == clusterer_alias:
                clusterer_dict[c_name] = True
            elif c_name not in clusterer_dict:
                clusterer_dict[c_name] = False
