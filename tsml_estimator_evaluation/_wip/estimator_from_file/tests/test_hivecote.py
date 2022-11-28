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

    assert _assert_array_almost_equal(probas[0], np.array([0.5, 0.5]), decimal=4)


def test_hivecote_from_file_check_estimator():
    """TODO"""
    check_estimator(FromFileHIVECOTE, return_exceptions=False)
