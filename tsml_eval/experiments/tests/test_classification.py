# -*- coding: utf-8 -*-
"""Tests for classification experiments."""

__author__ = ["MatthewMiddlehurst"]

import os

import pytest

from tsml_eval.experiments import set_classifier
from tsml_eval.experiments.classification_experiments import run_experiment
from tsml_eval.utils.test_utils import EXEMPT_ESTIMATOR_NAMES, _check_set_method
from tsml_eval.utils.tests.test_results_writing import _check_classification_file_format


@pytest.mark.parametrize(
    "classifier",
    ["DummyClassifier-tsml", "DummyClassifier-aeon", "DummyClassifier-sklearn"],
)
@pytest.mark.parametrize(
    "dataset",
    ["MinimalChinatown", "UnequalMinimalChinatown", "EqualMinimalJapaneseVowels"],
)
def test_run_classification_experiment(classifier, dataset):
    """Test classification experiments with test data and classifier."""
    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../datasets/"
    )
    result_path = (
        "./test_output/classification/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/classification/"
    )

    args = [
        None,
        data_path,
        result_path,
        classifier,
        dataset,
        "0",
        "True",
        "False",
    ]

    # aeon estimators don't support unequal length series lists currently
    try:
        run_experiment(args, overwrite=True)
    except ValueError as e:
        if "not support unequal length series" in str(e):
            return
        else:
            raise e

    test_file = f"{result_path}{classifier}/Predictions/{dataset}/testResample0.csv"
    train_file = f"{result_path}{classifier}/Predictions/{dataset}/trainResample0.csv"

    assert os.path.exists(test_file) and os.path.exists(train_file)

    _check_classification_file_format(test_file)
    _check_classification_file_format(train_file)

    os.remove(test_file)
    os.remove(train_file)


def test_set_classifier():
    """Test set_classifier method."""
    classifier_lists = [
        set_classifier.convolution_based_classifiers,
        set_classifier.deep_learning_classifiers,
        set_classifier.dictionary_based_classifiers,
        set_classifier.distance_based_classifiers,
        set_classifier.feature_based_classifiers,
        set_classifier.hybrid_classifiers,
        set_classifier.interval_based_classifiers,
        set_classifier.other_classifiers,
        set_classifier.shapelet_based_classifiers,
        set_classifier.vector_classifiers,
    ]

    classifier_dict = {}
    all_classifier_names = []

    for classifier_list in classifier_lists:
        _check_set_method(
            set_classifier.set_classifier,
            classifier_list,
            classifier_dict,
            all_classifier_names,
        )

    for estimator in EXEMPT_ESTIMATOR_NAMES:
        if estimator in classifier_dict:
            classifier_dict.pop(estimator)

    if not all(classifier_dict.values()):
        missing_keys = [key for key, value in classifier_dict.items() if not value]

        raise ValueError(
            "All classifiers seen in set_classifier must have an entry for the full "
            "class name (usually with default parameters). Classifiers with missing "
            f"entries: {missing_keys}."
        )
