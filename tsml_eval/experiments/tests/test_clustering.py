# -*- coding: utf-8 -*-
__author__ = ["MatthewMiddlehurst"]

import os

from tsml_eval.experiments.clustering_experiments import run_experiment


def test_run_experiment():
    result_path = (
        "./test_output/clustering/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/clustering/"
    )
    data_path = (
        "./tsml_eval/data/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../data/",
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

    test_file = (
        result_path + clusterer + "/Predictions/" + dataset + "/testResample0.csv"
    )
    train_file = (
        result_path + clusterer + "/Predictions/" + dataset + "/trainResample0.csv"
    )

    assert os.path.exists(test_file) and os.path.exists(train_file)

    os.remove(test_file)
    os.remove(train_file)
