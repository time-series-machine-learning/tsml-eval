"""Tests for publication experiments estimator selection."""

import pytest

from tsml_eval.publications.y2023.rist_pipeline import (
    _set_rist_classifier,
    _set_rist_regressor,
    rist_classifiers,
    rist_regressors,
)
from tsml_eval.testing.testing_utils import _check_set_method, _check_set_method_results


def test_set_rist_classifier():
    """Test set_rist_classifier method."""
    classifier_list = []
    classifier_dict = {}
    all_classifier_names = []
    _check_set_method(
        _set_rist_classifier,
        rist_classifiers,
        classifier_list,
        classifier_dict,
        all_classifier_names,
    )

    _check_set_method_results(
        classifier_dict,
        estimator_name="Classifiers",
        method_name="_set_rist_classifier",
    )


def test_set_rist_classifier_invalid():
    """Test set_rist_classifierr method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN CLASSIFIER"):
        _set_rist_classifier("invalid")


def test_set_rist_regressor():
    """Test set_rist_regressors method."""
    regressor_list = []
    regressor_dict = {}
    all_regressor_names = []
    _check_set_method(
        _set_rist_regressor,
        rist_regressors,
        regressor_list,
        regressor_dict,
        all_regressor_names,
    )

    _check_set_method_results(
        regressor_dict, estimator_name="Regressors", method_name="_set_rist_regressor"
    )


def test_set_rist_regressor_invalid():
    """Test set_rist_regressor method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN REGRESSOR"):
        _set_rist_regressor("invalid")
