"""Tests for publication experiments estimator selection."""

from tsml_eval.publications.y2023.tser_archive_expansion import (
    _set_tser_exp_regressor,
    expansion_regressors,
)
from tsml_eval.utils.test_utils import EXEMPT_ESTIMATOR_NAMES, _check_set_method


def test_set_expansion_regressor():
    """Test set_expansion_regressor method."""
    regressor_dict = {}
    all_regressor_names = []

    _check_set_method(
        _set_tser_exp_regressor,
        expansion_regressors,
        regressor_dict,
        all_regressor_names,
    )

    for estimator in EXEMPT_ESTIMATOR_NAMES:
        if estimator in regressor_dict:
            regressor_dict.pop(estimator)

    if not all(regressor_dict.values()):
        missing_keys = [key for key, value in regressor_dict.items() if not value]

        raise ValueError(
            "All regressors seen in _set_tser_exp_regressor must have an entry for the "
            "full class name (usually with default parameters). regressors with "
            f"missing entries: {missing_keys}."
        )
