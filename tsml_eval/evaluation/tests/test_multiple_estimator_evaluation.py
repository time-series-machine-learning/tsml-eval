from tsml_eval.evaluation.multiple_estimator_evaluation import (
    evaluate_classifiers_by_problem,
    evaluate_clusterers_by_problem,
    evaluate_forecasters_by_problem,
    evaluate_regressors_by_problem,
)
from tsml_eval.testing.test_utils import _TEST_OUTPUT_PATH, _TEST_RESULTS_PATH


def test_evaluate_classifiers_by_problem():
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
    classifiers = ["ROCKET", "TSF", "1NN-DTW"]
    datasets = ["Chinatown", "ItalyPowerDemand", "Trace"]
    resamples = 3

    evaluate_clusterers_by_problem(
        _TEST_RESULTS_PATH + "/clustering/",
        classifiers,
        datasets,
        _TEST_OUTPUT_PATH + "/eval/clustering/",
        resamples=resamples,
        eval_name="test0",
    )


def test_evaluate_regressors_by_problem():
    classifiers = ["ROCKET", "TSF", "1NN-DTW"]
    datasets = ["Chinatown", "ItalyPowerDemand", "Trace"]
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
    classifiers = ["ROCKET", "TSF"]
    datasets = ["Chinatown", "ItalyPowerDemand"]
    resamples = 1

    evaluate_forecasters_by_problem(
        _TEST_RESULTS_PATH + "/forecasting/",
        classifiers,
        datasets,
        _TEST_OUTPUT_PATH + "/eval/forecasting/",
        resamples=resamples,
        eval_name="test0",
    )
