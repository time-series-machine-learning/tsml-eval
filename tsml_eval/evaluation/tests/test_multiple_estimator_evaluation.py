"""Tests for the multiple estimator evaluation functionality."""

import os
import shutil

import numpy as np
import pytest

from tsml_eval.evaluation.multiple_estimator_evaluation import (
    evaluate_classifiers,
    evaluate_classifiers_by_problem,
    evaluate_classifiers_from_file,
    evaluate_clusterers_by_problem,
    evaluate_forecasters_by_problem,
    evaluate_regressors_by_problem,
)
from tsml_eval.evaluation.storage import ClassifierResults
from tsml_eval.testing.testing_utils import _TEST_OUTPUT_PATH, _TEST_RESULTS_PATH


def _read_rows(path):
    with open(path) as f:
        return {
            line.split(",")[0]: line.strip().split(",")[1:]
            for line in f
            if line.strip()
        }


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


def test_evaluate_classifiers_nan_statistic():
    """Test a NaN statistic does not silently alter the evaluation."""
    classifiers = ["ROCKET", "TSF", "1NN-DTW"]
    datasets = ["Chinatown", "ItalyPowerDemand", "Trace"]
    save_path = _TEST_OUTPUT_PATH + "/eval/classification/"
    eval_path = save_path + "nan_statistic/"
    shutil.rmtree(eval_path, ignore_errors=True)

    results = []
    names = []
    for classifier in classifiers:
        for dataset in datasets:
            cr = ClassifierResults().load_from_file(
                f"{_TEST_RESULTS_PATH}/classification/{classifier}/Predictions/"
                f"{dataset}/testResample0.csv"
            )
            if classifier == "TSF" and dataset == "Trace":
                cr.auroc_score = np.nan
            results.append(cr)
            names.append(classifier)

    with pytest.warns(UserWarning, match=r"test AUROC is NaN for datasets \['Trace'\]"):
        evaluate_classifiers(
            results, save_path, eval_name="nan_statistic", estimator_names=names
        )

    means = _read_rows(eval_path + "AUROC/auroc_mean.csv")
    tsf = means["Estimators:"].index("TSF")
    assert means["Trace"][tsf] == "nan"
    assert means["Trace"].count("nan") == 1

    ranks = _read_rows(eval_path + "AUROC/auroc_ranks.csv")
    assert ranks["Trace"] == ["nan"] * 3
    assert "nan" not in ranks["Chinatown"]

    summary = _read_rows(eval_path + "nan_statistic_summary.csv")
    assert summary["testAUROCMean"].count("nan") == 1
    assert summary["testAUROCAvgRank"] == ["nan"] * 3
    assert "nan" not in summary["testAccuracyAvgRank"]

    assert not os.path.exists(eval_path + "AUROC/auroc_p_values.csv")
    assert not os.path.exists(eval_path + "AUROC/figures/")
    assert os.path.exists(eval_path + "Accuracy/accuracy_p_values.csv")
    assert os.path.exists(eval_path + "Accuracy/figures/")


@pytest.mark.parametrize(
    "error_on_missing, continue_on_missing",
    [(True, False), (False, False), (False, True)],
)
def test_evaluate_classifiers_by_problem_broken_results(
    error_on_missing, continue_on_missing
):
    """Test results files which fail to load are treated as missing."""
    datasets = ["Chinatown", "Trace"]
    save_path = _TEST_OUTPUT_PATH + "/eval/classification/"
    eval_name = f"broken_{error_on_missing}_{continue_on_missing}"
    eval_path = f"{save_path}{eval_name}/"
    shutil.rmtree(eval_path, ignore_errors=True)

    kwargs = {
        "load_path": [
            _TEST_RESULTS_PATH + "/classification/",
            _TEST_RESULTS_PATH + "/broken/",
        ],
        "classifier_names": [["ROCKET", "TSF"], ["BrokenROCKET"]],
        "dataset_names": [datasets, datasets],
        "save_path": save_path,
        "error_on_missing": error_on_missing,
        "continue_on_missing": continue_on_missing,
        "eval_name": eval_name,
    }

    if error_on_missing:
        with pytest.raises(
            ValueError,
            match="BrokenROCKET on Chinatown test resample 0 could not be loaded. "
            "ValueError: AUROC is undefined when fewer than two classes",
        ):
            evaluate_classifiers_by_problem(**kwargs)
        return

    evaluate_classifiers_by_problem(**kwargs)

    means = _read_rows(eval_path + "Accuracy/accuracy_mean.csv")
    if continue_on_missing:
        broken = means["Estimators:"].index("BrokenROCKET")
        assert means["Chinatown"][broken] == "nan"
        assert not os.path.exists(eval_path + f"{eval_name}_summary.csv")
    else:
        assert "Chinatown" not in means
        assert "Trace" in means
        assert os.path.exists(eval_path + f"{eval_name}_summary.csv")


@pytest.mark.parametrize("error_on_missing", [True, False])
def test_evaluate_classifiers_from_file_broken_results(error_on_missing):
    """Test results files which fail to load are skipped along with their names."""
    paths = {
        "TSF": _TEST_RESULTS_PATH + "/classification/",
        "BrokenROCKET": _TEST_RESULTS_PATH + "/broken/",
        "1NN-DTW": _TEST_RESULTS_PATH + "/classification/",
    }
    datasets = ["Chinatown", "Trace"]
    save_path = _TEST_OUTPUT_PATH + "/eval/classification/"
    eval_name = f"broken_from_file_{error_on_missing}"
    eval_path = f"{save_path}{eval_name}/"
    shutil.rmtree(eval_path, ignore_errors=True)

    # the broken file is not last, so skipping it without its name would shift the
    # names of every result after it
    load_paths = [
        f"{path}{classifier}/Predictions/{dataset}/testResample0.csv"
        for classifier, path in paths.items()
        for dataset in datasets
    ]
    names = [classifier for classifier in paths for _ in datasets]

    kwargs = {
        "load_paths": load_paths,
        "save_path": save_path,
        "error_on_missing": error_on_missing,
        "continue_on_missing": True,
        "eval_name": eval_name,
        "estimator_names": names,
    }

    if error_on_missing:
        with pytest.raises(
            ValueError,
            match="BrokenROCKET/Predictions/Chinatown/testResample0.csv could not be "
            "loaded",
        ):
            evaluate_classifiers_from_file(**kwargs)
        return

    evaluate_classifiers_from_file(**kwargs)

    means = _read_rows(eval_path + "Accuracy/accuracy_mean.csv")
    assert means["Estimators:"] == ["1NN-DTW", "BrokenROCKET", "TSF"]
    assert means["Chinatown"][1] == "nan"
    for i, classifier in enumerate(means["Estimators:"]):
        cr = ClassifierResults().load_from_file(
            f"{paths[classifier]}{classifier}/Predictions/Trace/testResample0.csv"
        )
        assert float(means["Trace"][i]) == pytest.approx(cr.accuracy)


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
