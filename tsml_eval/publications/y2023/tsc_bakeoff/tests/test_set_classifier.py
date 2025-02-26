"""Tests for publication experiments estimator selection."""

import pytest

from tsml_eval.publications.y2023.tsc_bakeoff import (
    _set_bakeoff_classifier,
    bakeoff_classifiers,
)
from tsml_eval.testing.testing_utils import _check_set_method, _check_set_method_results


def test_set_bakeoff_classifier():
    """Test set_bakeoff_classifier method."""
    classifier_list = []
    classifier_dict = {}
    all_classifier_names = []
    _check_set_method(
        _set_bakeoff_classifier,
        bakeoff_classifiers,
        classifier_list,
        classifier_dict,
        all_classifier_names,
    )

    _check_set_method_results(
        classifier_dict,
        estimator_name="Classifiers",
        method_name="_set_bakeoff_classifier",
    )


def test_set_bakeoff_classifier_invalid():
    """Test set_bakeoff_classifier method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN CLASSIFIER"):
        _set_bakeoff_classifier("invalid")
