"""Tests for publication experiments estimator selection."""

import pytest

from tsml_eval.publications.y2023.tser_archive_expansion import (
    _set_tser_exp_regressor,
    expansion_regressors,
)
from tsml_eval.testing.testing_utils import _check_set_method, _check_set_method_results


def test_set_expansion_regressor():
    """Test set_tser_exp_regressor method."""
    regressor_list = []
    regressor_dict = {}
    all_regressor_names = []
    _check_set_method(
        _set_tser_exp_regressor,
        expansion_regressors,
        regressor_list,
        regressor_dict,
        all_regressor_names,
    )

    _check_set_method_results(
        regressor_dict,
        estimator_name="Regressors",
        method_name="_set_tser_exp_regressor",
    )


def test_set_expansion_regressor_invalid():
    """Test set_tser_exp_regressor method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN REGRESSOR"):
        _set_tser_exp_regressor("invalid")
