# -*- coding: utf-8 -*-
"""Tests for building HIVE-COTE from file."""

__author__ = ["MatthewMiddlehurst"]

import numpy as np
from aeon.datasets import load_arrow_head, load_italy_power_demand
from aeon.utils._testing.estimator_checks import _assert_array_almost_equal
from aeon.utils.estimator_checks import check_estimator

from tsml_eval._wip.estimator_from_file.hivecote import FromFileHIVECOTE


def test_hivecote_from_file():
    """Test HIVE-COTE from file with ItalyPowerDemand results."""
    X_train, y_train = load_italy_power_demand(split="train")
    X_test, _ = load_italy_power_demand(split="test")

    file_paths = [
        "test_files/ItalyPowerDemand/Arsenal/",
        "test_files/ItalyPowerDemand/DrCIF/",
        "test_files/ItalyPowerDemand/STC/",
        "test_files/ItalyPowerDemand/TDE/",
    ]

    hc2 = FromFileHIVECOTE(file_paths=file_paths, random_state=0)
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
    _assert_array_almost_equal(probas[0], np.array([0.0785, 0.9215]), decimal=4)


def test_tuned_hivecote_from_file():
    """Test HIVE-COTE from file tuned alpha with ArrowHead results."""
    X_train, y_train = load_arrow_head(split="train")
    X_test, _ = load_arrow_head(split="test")

    file_paths = [
        "test_files/ArrowHead/Arsenal/",
        "test_files/ArrowHead/DrCIF/",
        "test_files/ArrowHead/STC/",
        "test_files/ArrowHead/TDE/",
    ]

    hc2 = FromFileHIVECOTE(file_paths=file_paths, tune_alpha=True, random_state=0)
    hc2.fit(X_train, y_train)
    probas = hc2.predict_proba(X_test)

    assert probas.shape == (X_test.shape[0], 3)
    _assert_array_almost_equal(probas[0], np.array([0.6092, 0.2308, 0.16]), decimal=4)


def test_hivecote_from_file_check_estimator():
    """Test HIVE-COTE meets the aeon estimator interface."""
    check_estimator(FromFileHIVECOTE, return_exceptions=False)
