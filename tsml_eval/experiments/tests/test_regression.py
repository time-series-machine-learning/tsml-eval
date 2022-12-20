# -*- coding: utf-8 -*-
"""Tests for regression utilities."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os

from tsml_eval.experiments.regression_experiments import run_experiment


def test_run_experiment():
    result_path = (
        "./test_output/regression/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/regression/"
    )
    data_path = (
        "./tsml_eval/data/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../data/",
    )
    regressor = "DummyRegressor"
    dataset = "Covid3Month"
    args = [
        None,
        dataset,
        result_path,
        regressor,
        dataset,
        "1",
        "True",
        "False",
    ]
    run_experiment(args, overwrite=True)

    test_file = (
        result_path + regressor + "/Predictions/" + dataset + "/testResample0.csv"
    )
    train_file = (
        result_path + regressor + "/Predictions/" + dataset + "/trainResample0.csv"
    )

    assert os.path.exists(test_file) and os.path.exists(train_file)

    os.remove(test_file)
    os.remove(train_file)
