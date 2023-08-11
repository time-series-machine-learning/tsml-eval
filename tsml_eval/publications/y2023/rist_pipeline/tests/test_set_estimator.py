"""Tests for publication experiments estimator selection."""

from tsml_eval.publications.y2023.rist_pipeline import (
    _set_rist_classifier,
    _set_rist_regressor,
    rist_classifiers,
    rist_regressors,
)
from tsml_eval.utils.test_utils import EXEMPT_ESTIMATOR_NAMES, _check_set_method


def test_set_rist_classifiers():
    """Test set_rist_classifier method."""
    classifier_dict = {}
    all_classifier_names = []

    _check_set_method(
        _set_rist_classifier,
        rist_classifiers,
        classifier_dict,
        all_classifier_names,
    )

    for estimator in EXEMPT_ESTIMATOR_NAMES:
        if estimator in classifier_dict:
            classifier_dict.pop(estimator)

    if not all(classifier_dict.values()):
        missing_keys = [key for key, value in classifier_dict.items() if not value]

        raise ValueError(
            "All classifiers seen in _set_rist_classifier must have an entry for "
            "the full class name (usually with default parameters). classifiers with "
            f"missing entries: {missing_keys}."
        )


def test_set_rist_regressors():
    """Test set_rist_regressors method."""
    regressors = {}
    all_regressor_names = []

    _check_set_method(
        _set_rist_regressor,
        rist_regressors,
        regressors,
        all_regressor_names,
    )

    for estimator in EXEMPT_ESTIMATOR_NAMES:
        if estimator in regressors:
            regressors.pop(estimator)

    if not all(regressors.values()):
        missing_keys = [key for key, value in regressors.items() if not value]

        raise ValueError(
            "All classifiers seen in _set_rist_regressor must have an entry for "
            "the full class name (usually with default parameters). classifiers with "
            f"missing entries: {missing_keys}."
        )
