# -*- coding: utf-8 -*-
"""Tests for publication experiments estimator selection."""

from tsml_eval.publications._2023.tser_archive_expansion import set_expansion_regressor
from tsml_eval.utils.test_utils import EXEMPT_ESTIMATOR_NAMES, _check_set_method


def test_set_expansion_regressor():
    """Test set_expansion_regressor method."""
    regressor_dict = {}
    all_regressor_names = []

    _check_set_method(
        set_expansion_regressor._set_tser_exp_regressor,
        set_expansion_regressor.expansion_regressors,
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
