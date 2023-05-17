# -*- coding: utf-8 -*-
"""Tests for regression experiments."""

__author__ = ["MatthewMiddlehurst"]

import os

import pytest

from tsml_eval.experiments import set_regressor
from tsml_eval.experiments.regression_experiments import run_experiment
from tsml_eval.utils.test_utils import EXEMPT_ESTIMATOR_NAMES, _check_set_method
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
    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../datasets/"
    )
    result_path = (
        "./test_output/regression/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/regression/"
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


def test_set_regressor():
    """Test set_regressor method."""
    regressor_lists = [
        set_regressor.convolution_based_regressors,
        set_regressor.deep_learning_regressors,
        set_regressor.dictionary_based_regressors,
        set_regressor.distance_based_regressors,
        set_regressor.feature_based_regressors,
        set_regressor.hybrid_regressors,
        set_regressor.interval_based_regressors,
        set_regressor.other_regressors,
        set_regressor.shapelet_based_regressors,
        set_regressor.vector_regressors,
    ]

    regressor_dict = {}
    all_regressor_names = []

    for regressor_list in regressor_lists:
        _check_set_method(
            set_regressor.set_regressor,
            regressor_list,
            regressor_dict,
            all_regressor_names,
        )

    for estimator in EXEMPT_ESTIMATOR_NAMES:
        if estimator in regressor_dict:
            regressor_dict.pop(estimator)

    if not all(regressor_dict.values()):
        missing_keys = [key for key, value in regressor_dict.items() if not value]

        raise ValueError(
            "All regressors seen in set_regressor must have an entry for the full "
            "class name (usually with default parameters). regressors with missing "
            f"entries: {missing_keys}."
        )
