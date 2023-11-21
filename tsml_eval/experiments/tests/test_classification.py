"""Tests for classification experiments."""

__author__ = ["MatthewMiddlehurst"]

import os
import runpy

import pytest
from tsml.dummy import DummyRegressor

from tsml_eval.datasets._test_data._data_sizes import DATA_TEST_SIZES
from tsml_eval.experiments import (
    classification_experiments,
    run_classification_experiment,
    set_classifier,
    threaded_classification_experiments,
)
from tsml_eval.experiments.tests import _CLASSIFIER_RESULTS_PATH
from tsml_eval.testing.test_utils import (
    _TEST_DATA_PATH,
    _check_set_method,
    _check_set_method_results,
)
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
    args = [
        _TEST_DATA_PATH,
        _CLASSIFIER_RESULTS_PATH,
        classifier,
        dataset,
        "0",
        "-tr",
    ]

    classification_experiments.run_experiment(args)

    test_file = (
        f"{_CLASSIFIER_RESULTS_PATH}{classifier}/Predictions/{dataset}/"
        "testResample0.csv"
    )
    train_file = (
        f"{_CLASSIFIER_RESULTS_PATH}{classifier}/Predictions/{dataset}/"
        "trainResample0.csv"
    )

    assert os.path.exists(test_file) and os.path.exists(train_file)

    _check_classification_file_format(
        test_file, num_results_lines=DATA_TEST_SIZES[dataset]
    )
    _check_classification_file_format(
        train_file, num_results_lines=DATA_TEST_SIZES[dataset]
    )

    # test present results checking
    classification_experiments.run_experiment(args)

    os.remove(test_file)
    os.remove(train_file)


def test_run_classification_experiment_main():
    """Test classification experiments main with test data and classifier."""
    classifier = "ROCKET"
    dataset = "MinimalChinatown"

    # run twice to test results present check
    for _ in range(2):
        runpy.run_path(
            "./tsml_eval/experiments/classification_experiments.py"
            if os.getcwd().split("\\")[-1] != "tests"
            else "../classification_experiments.py",
            run_name="__main__",
        )

    test_file = (
        f"{_CLASSIFIER_RESULTS_PATH}{classifier}/Predictions/{dataset}/"
        "testResample0.csv"
    )
    assert os.path.exists(test_file)
    _check_classification_file_format(test_file)

    os.remove(test_file)


def test_run_threaded_classification_experiment():
    """Test threaded classification experiments with test data and classifier."""
    classifier = "ROCKET"
    dataset = "MinimalChinatown"

    args = [
        _TEST_DATA_PATH,
        _CLASSIFIER_RESULTS_PATH,
        classifier,
        dataset,
        "1",
        "-nj",
        "2",
        # also test normalisation and benchmark time here
        "--row_normalise",
        "--benchmark_time",
    ]

    threaded_classification_experiments.run_experiment(args)

    test_file = (
        f"{_CLASSIFIER_RESULTS_PATH}{classifier}/Predictions/{dataset}/"
        "testResample1.csv"
    )
    assert os.path.exists(test_file)
    _check_classification_file_format(test_file)

    # test present results checking
    threaded_classification_experiments.run_experiment(args)

    # this covers the main method and experiment function result file checking
    runpy.run_path(
        "./tsml_eval/experiments/threaded_classification_experiments.py"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../threaded_classification_experiments.py",
        run_name="__main__",
    )

    os.remove(test_file)


def test_run_classification_experiment_invalid_build_settings():
    """Test run_classification_experiment method with invalid build settings."""
    with pytest.raises(ValueError, match="Both test_file and train_file"):
        run_classification_experiment(
            [],
            [],
            [],
            [],
            None,
            "",
            build_test_file=False,
            build_train_file=False,
        )


def test_run_classification_experiment_invalid_estimator():
    """Test run_classification_experiment method with invalid estimator."""
    with pytest.raises(TypeError, match="classifier must be a"):
        run_classification_experiment(
            [],
            [],
            [],
            [],
            DummyRegressor(),
            "",
        )


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

    _check_set_method_results(
        classifier_dict, estimator_name="Classifiers", method_name="set_classifier"
    )


def test_set_classifier_invalid():
    """Test set_classifier method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN CLASSIFIER"):
        set_classifier.set_classifier("invalid")
