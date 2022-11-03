# -*- coding: utf-8 -*-
from ..results_by_classifier import (
    get_single_classifier_results,
    get_single_classifier_results_from_web,
    valid_multi_classifiers,
    valid_uni_classifiers,
)


def test_load_local_results():
    """Test that we can load all the classifiers in valid_uni_classifiers and
    valid_multi_classifiers from the local directory results/tsml/ByClassifier. Add
    asserts later"""
    for cls in valid_uni_classifiers:
        get_single_classifier_results(cls)
    for cls in valid_multi_classifiers:
        get_single_classifier_results(cls, type="Multivariate")


def test_load_web_results():
    """Test that we can load all the classifiers in valid_uni_classifiers and
    valid_multi_classifiers from timerseriesclassification.com. Note mail fail for
    connection errors with the website."""
    for cls in valid_uni_classifiers:
        get_single_classifier_results_from_web(cls)
    for cls in valid_multi_classifiers:
        get_single_classifier_results_from_web(cls, type="Multivariate")
