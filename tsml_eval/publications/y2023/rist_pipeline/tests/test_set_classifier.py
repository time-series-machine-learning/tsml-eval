"""Tests for publication experiments estimator selection."""

from tsml_eval.publications.y2023.tsc_bakeoff import (
    _set_bakeoff_classifier,
    bakeoff_classifiers,
)
from tsml_eval.utils.test_utils import EXEMPT_ESTIMATOR_NAMES, _check_set_method


def test_set_bakeoff_classifiers():
    """Test set_bakeoff_classifier method."""
    classifier_dict = {}
    all_classifier_names = []

    _check_set_method(
        _set_bakeoff_classifier,
        bakeoff_classifiers,
        classifier_dict,
        all_classifier_names,
    )

    for estimator in EXEMPT_ESTIMATOR_NAMES:
        if estimator in classifier_dict:
            classifier_dict.pop(estimator)

    if not all(classifier_dict.values()):
        missing_keys = [key for key, value in classifier_dict.items() if not value]

        raise ValueError(
            "All classifiers seen in _set_bakeoff_classifier must have an entry for "
            "the full class name (usually with default parameters). classifiers with "
            f"missing entries: {missing_keys}."
        )
