"""Tests for the results loading utilities."""

import numpy as np
import pytest

from tsml_eval.evaluation.storage import (
    ClassifierResults,
    ClustererResults,
    ForecasterResults,
    RegressorResults,
)
from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH
from tsml_eval.utils.results_loading import (
    estimator_results_to_array,
    estimator_results_to_dict,
    load_estimator_results,
    load_estimator_results_to_array,
    load_estimator_results_to_dict,
)


@pytest.mark.parametrize(
    "type,path",
    [
        (
            ClassifierResults,
            _TEST_RESULTS_PATH
            + "/classification/ROCKET/Predictions/MinimalChinatown/testResample0.csv",
        ),
        (
            ClustererResults,
            _TEST_RESULTS_PATH
            + "/clustering/KMeans/Predictions/MinimalChinatown/trainResample0.csv",
        ),
        (
            RegressorResults,
            _TEST_RESULTS_PATH
            + "/regression/ROCKET/Predictions/MinimalGasPrices/testResample0.csv",
        ),
        (
            ForecasterResults,
            _TEST_RESULTS_PATH
            + "/forecasting/NaiveForecaster/Predictions/ShampooSales/testResample0.csv",
        ),
    ],
)
def test_load_estimator_results(type, path):
    """Test loading arbitrary estimator results from a file."""
    er = load_estimator_results(path)
    assert isinstance(er, type)


def test_load_estimator_results_invalid():
    """Test loading invalid estimator results from a file."""
    with pytest.raises(ValueError, match="Unable to determine the type"):
        load_estimator_results(_TEST_RESULTS_PATH + "/broken/brokenResultsFile.csv")


classifiers = ["ROCKET", "TSF", "1NN-DTW"]
datasets = ["Chinatown", "ItalyPowerDemand", "Trace"]


def test_load_estimator_results_to_dict():
    """Test loading results returned in a dict."""
    res = load_estimator_results_to_dict(
        _TEST_RESULTS_PATH + "/classification/",
        classifiers,
        datasets,
        "accuracy",
        resamples=1,
    )
    assert isinstance(res, dict)
    assert len(res) == 3
    assert all(len(v) == 3 for v in res.values())
    assert res["ROCKET"]["Chinatown"] == 0.9795918367346939

    # test resamples
    res2 = load_estimator_results_to_dict(
        _TEST_RESULTS_PATH + "/classification/",
        classifiers,
        datasets,
        "accuracy",
        resamples=3,
    )
    assert isinstance(res2, dict)
    assert len(res2) == 3
    assert all(len(v) == 3 for v in res2.values())
    assert isinstance(res2["ROCKET"]["Chinatown"], np.ndarray)
    assert len(res2["ROCKET"]["Chinatown"]) == 3
    assert res2["ROCKET"]["Chinatown"][0] == 0.9795918367346939
    assert np.average(res2["ROCKET"]["Chinatown"]) == 0.9698736637512148


def test_load_estimator_results_to_array():
    """Test loading results returned in an array."""
    res, data_names, est_names = load_estimator_results_to_array(
        _TEST_RESULTS_PATH + "/classification/",
        classifiers,
        datasets,
        "accuracy",
        resamples=1,
    )
    assert isinstance(res, np.ndarray)
    assert res.shape == (3, 3)
    assert res[0][0] == 0.9795918367346939
    assert isinstance(data_names, list) and isinstance(est_names, list)
    assert data_names == datasets
    assert est_names == classifiers

    # test resamples
    res2, data_names2, est_names2 = load_estimator_results_to_array(
        _TEST_RESULTS_PATH + "/classification/",
        classifiers,
        datasets,
        "accuracy",
        resamples=3,
    )
    assert isinstance(res2, np.ndarray)
    assert res2.shape == (3, 3)
    assert res2[0][0] == 0.9698736637512148
    assert isinstance(data_names2, list) and isinstance(est_names2, list)
    assert data_names2 == datasets
    assert est_names2 == classifiers


def test_load_estimator_results_missing():
    """Test loading results with missing results."""
    paths = [
        _TEST_RESULTS_PATH
        + "/classification/ROCKET/Predictions/Chinatown/testResample0.csv",
        _TEST_RESULTS_PATH
        + "/classification/TSF/Predictions/Chinatown/testResample0.csv",
        _TEST_RESULTS_PATH
        + "/classification/ROCKET/Predictions/Trace/testResample0.csv",
        _TEST_RESULTS_PATH + "/classification/TSF/Predictions/Trace/testResample0.csv",
        _TEST_RESULTS_PATH
        + "/classification/ROCKET/Predictions/ItalyPowerDemand/testResample0.csv",
    ]
    res_objects = [load_estimator_results(p) for p in paths]

    res_dict = estimator_results_to_dict(
        res_objects,
        "accuracy",
    )
    assert isinstance(res_dict, dict)
    assert len(res_dict) == 2
    assert len(res_dict["ROCKET"]) == 3
    assert len(res_dict["TSF"]) == 2

    res_arr, data_names, est_names = estimator_results_to_array(
        res_objects,
        "accuracy",
        include_missing=True,
    )
    assert isinstance(res_arr, np.ndarray)
    assert res_arr.shape == (3, 2)
    assert np.isnan(res_arr).any()
    assert len(data_names) == 3
    assert len(est_names) == 2
