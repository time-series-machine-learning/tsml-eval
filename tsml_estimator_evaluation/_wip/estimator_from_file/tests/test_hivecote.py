# -*- coding: utf-8 -*-
"""TODO"""

__author__ = ["MatthewMiddlehurst", ""]

import numpy as np
from sktime.datasets import load_italy_power_demand
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal
from sktime.utils.estimator_checks import check_estimator

from tsml_estimator_evaluation._wip.estimator_from_file.hivecote import FromFileHIVECOTE


def test_hivecote_from_file():
    """TODO"""
    train_X, train_y = load_italy_power_demand(split="train")
    test_X, test_y = load_italy_power_demand(split="test")

    file_paths = [
        "test_files/Arsenal/",
        "test_files/DrCIF/",
        "test_files/STC/",
        "test_files/TDE/",
    ]

    hc2 = FromFileHIVECOTE(file_paths=file_paths, random_state=0)
    hc2.fit(train_X, train_y)
    probas = hc2.predict_proba(test_X)

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

    assert probas.shape == (test_X.shape[0], 2)
    _assert_array_almost_equal(probas[0], np.array([0.0785, 0.9215]), decimal=4)


def test_hivecote_from_file_check_estimator():
    """TODO"""
    check_estimator(FromFileHIVECOTE, return_exceptions=False)
