# -*- coding: utf-8 -*-
"""Tests for regression utilities."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os

from sktime.datasets import load_arrow_head, load_unit_test

from tsml_estimator_evaluation.experiments.regression_experiments import (
    resample,
    run_experiment,
)


def test_run_experiment():
    result_path = (
        "../../../test_output/regression/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/regression/"
    )
    regressor = "RocketRegressor"
    dataset = "Covid3Month"
    args = [
        None,
        "./tsml_estimator_evaluation/data/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../data/",
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


def test_resample():
    train_X, train_y = load_arrow_head(split="train")
    test_X, test_y = load_unit_test(split="train")

    train_size = train_y.size
    test_size = test_y.size

    train_X, train_y, test_X, test_y = resample(train_X, train_y, test_X, test_y, 1)

    assert train_y.size == train_size and test_y.size == test_size
