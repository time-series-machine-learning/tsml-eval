# -*- coding: utf-8 -*-
"""Tests for regression experiments."""

__author__ = ["MatthewMiddlehurst"]

import os

from tsml_eval.experiments.regression_experiments import run_experiment
from tsml_eval.utils.tests.test_results_writing import _check_regression_file_format


def test_run_regression_experiment():
    """Test regression experiments with test data and regressor."""
    result_path = (
        "./test_output/regression/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/regression/"
    )
    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../datasets/"
    )
    regressor = "DummyRegressor"
    dataset = "Covid3Month"

    args = [
        None,
        data_path,
        result_path,
        regressor,
        dataset,
        "1",
        "True",
        "False",
    ]
    run_experiment(args, overwrite=True)

    test_file = f"{result_path}{regressor}/Predictions/{dataset}/testResample0.csv"
    train_file = f"{result_path}{regressor}/Predictions/{dataset}/trainResample0.csv"

    assert os.path.exists(test_file) and os.path.exists(train_file)

    _check_regression_file_format(test_file)
    _check_regression_file_format(train_file)

    os.remove(test_file)
    os.remove(train_file)
