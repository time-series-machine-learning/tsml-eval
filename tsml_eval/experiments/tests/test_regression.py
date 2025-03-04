"""Tests for regression experiments."""

__maintainer__ = ["MatthewMiddlehurst"]

import os
import runpy

import pytest
from aeon.utils.discovery import all_estimators
from tsml.dummy import DummyClassifier

from tsml_eval.datasets._test_data._data_sizes import DATA_TEST_SIZES
from tsml_eval.experiments import (
    _get_regressor,
    get_regressor_by_name,
    regression_experiments,
    run_regression_experiment,
    threaded_regression_experiments,
)
from tsml_eval.experiments.tests import _REGRESSOR_RESULTS_PATH
from tsml_eval.testing.testing_utils import (
    _TEST_DATA_PATH,
    _check_set_method,
    _check_set_method_results,
)
from tsml_eval.utils.tests.test_results_writing import _check_regression_file_format


@pytest.mark.parametrize(
    "regressor",
    ["DummyRegressor-tsml", "DummyRegressor-aeon", "DummyRegressor-sklearn"],
)
@pytest.mark.parametrize(
    "dataset",
    ["MinimalGasPrices", "UnequalMinimalGasPrices", "MinimalCardanoSentiment"],
)
def test_run_regression_experiment(regressor, dataset):
    """Test regression experiments with test data and regressor."""
    if regressor == "DummyRegressor-aeon" and dataset == "UnequalMinimalGasPrices":
        return  # todo remove when aeon dummy supports unequal

    args = [
        _TEST_DATA_PATH,
        _REGRESSOR_RESULTS_PATH,
        regressor,
        dataset,
        "0",
        "-tr",
    ]

    regression_experiments.run_experiment(args)

    test_file = (
        f"{_REGRESSOR_RESULTS_PATH}{regressor}/Predictions/{dataset}/testResample0.csv"
    )
    train_file = (
        f"{_REGRESSOR_RESULTS_PATH}{regressor}/Predictions/{dataset}/trainResample0.csv"
    )

    assert os.path.exists(test_file) and os.path.exists(train_file)

    _check_regression_file_format(test_file, num_results_lines=DATA_TEST_SIZES[dataset])
    _check_regression_file_format(
        train_file, num_results_lines=DATA_TEST_SIZES[dataset]
    )

    # test present results checking
    regression_experiments.run_experiment(args)

    os.remove(test_file)
    os.remove(train_file)


def test_run_regression_experiment_main():
    """Test regression experiments main with test data and regressor."""
    regressor = "ROCKET"
    dataset = "MinimalGasPrices"

    # run twice to test results present check
    for _ in range(2):
        runpy.run_path(
            (
                "./tsml_eval/experiments/regression_experiments.py"
                if os.getcwd().split("\\")[-1] != "tests"
                else "../regression_experiments.py"
            ),
            run_name="__main__",
        )

    test_file = (
        f"{_REGRESSOR_RESULTS_PATH}{regressor}/Predictions/{dataset}/testResample0.csv"
    )
    assert os.path.exists(test_file)
    _check_regression_file_format(test_file)

    os.remove(test_file)


def test_run_threaded_regression_experiment():
    """Test threaded regression experiments with test data and regressor."""
    regressor = "ROCKET"
    dataset = "MinimalGasPrices"

    args = [
        _TEST_DATA_PATH,
        _REGRESSOR_RESULTS_PATH,
        regressor,
        dataset,
        "1",
        "-nj",
        "2",
        # also test transforms and benchmark time here
        "--row_normalise",
        "--data_transform_name",
        "Padder",
        "--benchmark_time",
    ]

    threaded_regression_experiments.run_experiment(args)

    test_file = (
        f"{_REGRESSOR_RESULTS_PATH}{regressor}/Predictions/{dataset}/testResample1.csv"
    )
    assert os.path.exists(test_file)
    _check_regression_file_format(test_file)

    # test present results checking
    threaded_regression_experiments.run_experiment(args)

    # this covers the main method and experiment function result file checking
    runpy.run_path(
        (
            "./tsml_eval/experiments/threaded_regression_experiments.py"
            if os.getcwd().split("\\")[-1] != "tests"
            else "../threaded_regression_experiments.py"
        ),
        run_name="__main__",
    )

    os.remove(test_file)


def test_run_regression_experiment_invalid_build_settings():
    """Test run_regression_experiment method with invalid build settings."""
    with pytest.raises(ValueError, match="Both test_file and train_file"):
        run_regression_experiment(
            [],
            [],
            [],
            [],
            None,
            "",
            build_test_file=False,
            build_train_file=False,
        )


def test_run_regression_experiment_invalid_estimator():
    """Test run_regression_experiment method with invalid estimator."""
    with pytest.raises(TypeError, match="regressor must be a"):
        run_regression_experiment(
            [],
            [],
            [],
            [],
            DummyClassifier(),
            "",
        )


def test_get_regressor_by_name():
    """Test get_regressor_by_name method."""
    regressor_name_lists = [
        _get_regressor.convolution_based_regressors,
        _get_regressor.deep_learning_regressors,
        _get_regressor.distance_based_regressors,
        _get_regressor.feature_based_regressors,
        _get_regressor.hybrid_regressors,
        _get_regressor.interval_based_regressors,
        _get_regressor.other_regressors,
        _get_regressor.shapelet_based_regressors,
        _get_regressor.vector_regressors,
    ]

    regressor_list = []
    regressor_dict = {}
    all_regressor_names = []
    for regressor_name_list in regressor_name_lists:
        _check_set_method(
            get_regressor_by_name,
            regressor_name_list,
            regressor_list,
            regressor_dict,
            all_regressor_names,
        )

    _check_set_method_results(
        regressor_dict, estimator_name="Regressors", method_name="get_regressor_by_name"
    )


def test_get_regressor_by_name_invalid():
    """Test get_regressor_by_name method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN REGRESSOR"):
        get_regressor_by_name("invalid")


def test_aeon_regressors_available():
    """Test all aeon regressors are available."""
    excluded = [
        # composable/wrapper
        "RegressorPipeline",
        "RegressorEnsemble",
        "SklearnRegressorWrapper",
        "IntervalForestRegressor",
        # just missing
    ]

    est = [e for e, _ in all_estimators(type_filter="regressor")]
    for e in est:
        if e in excluded:
            continue

        try:
            assert get_regressor_by_name(e) is not None
        except ModuleNotFoundError:
            continue
