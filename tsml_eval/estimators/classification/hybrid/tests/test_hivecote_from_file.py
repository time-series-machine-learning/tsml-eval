"""Tests for building HIVE-COTE from file."""

__maintainer__ = ["MatthewMiddlehurst"]


import numpy as np
from aeon.datasets import load_arrow_head, load_italy_power_demand
from numpy.testing import assert_array_almost_equal

from tsml_eval.estimators.classification.hybrid.hivecote_from_file import (
    FromFileHIVECOTE,
)
from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH


def test_hivecote_from_file():
    """Test HIVE-COTE from file with ItalyPowerDemand results."""
    X_train, y_train = load_italy_power_demand(split="train")
    X_test, _ = load_italy_power_demand(split="test")

    file_paths = [
        _TEST_RESULTS_PATH + "/classification/Arsenal/Predictions/ItalyPowerDemand/",
        _TEST_RESULTS_PATH + "/classification/DrCIF/Predictions/ItalyPowerDemand/",
        _TEST_RESULTS_PATH + "/classification/STC/Predictions/ItalyPowerDemand/",
        _TEST_RESULTS_PATH + "/classification/TDE/Predictions/ItalyPowerDemand/",
    ]

    hc2 = FromFileHIVECOTE(classifiers=file_paths, random_state=0)
    hc2.fit(X_train, y_train)
    probas = hc2.predict_proba(X_test)

    # Component weights (a=4):
    # TDE: 0.941621858
    # STC: 0.88583781517
    # DrCIF: 0.8325698605
    # Arsenal: 0.8325698605

    # First case test probabilities:
    # TDE: 0.2091581219241968, 0.7908418780758033
    # STC: 0.025, 0.975
    # DrCIF: 0.066, 0.934
    # Arsenal: 0.0, 1.0

    # Weighted probabilities:
    # TDE: 0.19694785938, 0.74467399861
    # STC: 0.02214594537, 0.86369186979
    # DrCIF: 0.05494961079 , 0.7776202497
    # Arsenal: 0.0, 0.8325698605

    # Sum of weighted probabilities:
    # 0.27404341554, 3.2185559786

    # Normalised weighted probabilities:
    # 0.07846402768, 0.92153597231

    assert probas.shape == (X_test.shape[0], 2)
    assert_array_almost_equal(probas[0], np.array([0.0785, 0.9215]), decimal=4)


def test_tuned_hivecote_from_file():
    """Test HIVE-COTE from file tuned alpha with ArrowHead results."""
    X_train, y_train = load_arrow_head(split="train")
    X_test, _ = load_arrow_head(split="test")

    file_paths = [
        _TEST_RESULTS_PATH + "/classification/Arsenal/Predictions/ArrowHead/",
        _TEST_RESULTS_PATH + "/classification/DrCIF/Predictions/ArrowHead/",
        _TEST_RESULTS_PATH + "/classification/STC/Predictions/ArrowHead/",
        _TEST_RESULTS_PATH + "/classification/TDE/Predictions/ArrowHead/",
    ]

    hc2 = FromFileHIVECOTE(classifiers=file_paths, tune_alpha=True, random_state=0)
    hc2.fit(X_train, y_train)
    probas = hc2.predict_proba(X_test)

    assert probas.shape == (X_test.shape[0], 3)
    assert_array_almost_equal(probas[0], np.array([0.6092, 0.2308, 0.16]), decimal=4)
