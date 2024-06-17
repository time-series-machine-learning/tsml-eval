"""Tests for the multiple estimator evaluation functionality."""

from tsml_eval.evaluation.multiple_estimator_evaluation import (
    evaluate_classifiers_by_problem,
    evaluate_clusterers_by_problem,
    evaluate_forecasters_by_problem,
    evaluate_regressors_by_problem,
)
from tsml_eval.testing.testing_utils import _TEST_OUTPUT_PATH, _TEST_RESULTS_PATH


def test_evaluate_classifiers_by_problem():
    """Test the evaluation of classifiers by problem."""
    classifiers = ["ROCKET", "TSF", "1NN-DTW"]
    datasets = ["Chinatown", "ItalyPowerDemand", "Trace"]
    resamples = 3

    evaluate_classifiers_by_problem(
        _TEST_RESULTS_PATH + "/classification/",
        classifiers,
        datasets,
        _TEST_OUTPUT_PATH + "/eval/classification/",
        resamples=resamples,
        eval_name="test0",
    )


def test_evaluate_clusterers_by_problem():
    """Test the evaluation of clusterers by problem."""
    classifiers = ["KMeans", "KMeans-dtw", "KMeans-msm"]
    datasets = ["Chinatown", "ItalyPowerDemand", "Trace"]
    resamples = 3

    evaluate_clusterers_by_problem(
        _TEST_RESULTS_PATH + "/clustering/",
        classifiers,
        datasets,
        _TEST_OUTPUT_PATH + "/eval/clustering/",
        resamples=resamples,
        load_test_results=False,
        eval_name="test0",
    )


def test_evaluate_regressors_by_problem():
    """Test the evaluation of regressors by problem."""
    classifiers = ["ROCKET", "TSF", "1NN-DTW"]
    datasets = ["Covid3Month", "NaturalGasPricesSentiment", "FloodModeling1"]
    resamples = 3

    evaluate_regressors_by_problem(
        _TEST_RESULTS_PATH + "/regression/",
        classifiers,
        datasets,
        _TEST_OUTPUT_PATH + "/eval/regression/",
        resamples=resamples,
        eval_name="test0",
    )


def test_evaluate_forecasters_by_problem():
    """Test the evaluation of forecasters by problem."""
    classifiers = ["NaiveForecaster", "RandomForest", "LinearRegression"]
    datasets = ["Airline", "ShampooSales"]
    resamples = 1

    evaluate_forecasters_by_problem(
        _TEST_RESULTS_PATH + "/forecasting/",
        classifiers,
        datasets,
        _TEST_OUTPUT_PATH + "/eval/forecasting/",
        resamples=resamples,
        eval_name="test0",
    )
