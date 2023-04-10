# -*- coding: utf-8 -*-
"""Tests for regression experiments."""

__author__ = ["MatthewMiddlehurst"]

import os

import pytest

from tsml_eval.experiments.regression_experiments import run_experiment
from tsml_eval.utils.tests.test_results_writing import _check_regression_file_format


@pytest.mark.parametrize(
    "regressor",
    ["DummyRegressor-tsml", "DummyRegressor-aeon", "DummyRegressor-sklearn"],
)
@pytest.mark.parametrize(
    "dataset",
    ["MinimalGasPrices", "UnequalMinimalGasPrices"],
)
def test_run_regression_experiment(regressor, dataset):
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

    args = [
        None,
        data_path,
        result_path,
        regressor,
        dataset,
        "0",
        "True",
        "False",
        None,
    ]

    # aeon estimators don't support unequal length series lists currently
    try:
        run_experiment(args, overwrite=True)
    except ValueError as e:
        if "not support unequal length series" in str(e):
            return
        else:
            raise e

    test_file = f"{result_path}{regressor}/Predictions/{dataset}/testResample0.csv"
    train_file = f"{result_path}{regressor}/Predictions/{dataset}/trainResample0.csv"

    assert os.path.exists(test_file) and os.path.exists(train_file)

    _check_regression_file_format(test_file)
    _check_regression_file_format(train_file)

    os.remove(test_file)
    os.remove(train_file)
