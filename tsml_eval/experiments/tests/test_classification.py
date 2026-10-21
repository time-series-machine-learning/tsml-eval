"""Tests for classification experiments."""

import os
import runpy

import pytest
from aeon.regression import DummyRegressor
from aeon.utils.discovery import all_estimators

from tsml_eval.datasets._test_data._data_sizes import DATA_TEST_SIZES
from tsml_eval.experiments import (
    _get_classifier,
    classification_experiments,
    get_classifier_by_name,
    run_classification_experiment,
    threaded_classification_experiments,
)
from tsml_eval.experiments.tests import _CLASSIFIER_RESULTS_PATH
from tsml_eval.testing.testing_utils import (
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
    if classifier == "DummyClassifier-tsml":
        pytest.importorskip("tsml")

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
            (
                "./tsml_eval/experiments/classification_experiments.py"
                if os.getcwd().split("\\")[-1] != "tests"
                else "../classification_experiments.py"
            ),
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
        # also test transforms and benchmark time here
        "--row_normalise",
        "--data_transform_name",
        "Truncate-max",
        "--data_transform_name",
        "Padder",
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
        (
            "./tsml_eval/experiments/threaded_classification_experiments.py"
            if os.getcwd().split("\\")[-1] != "tests"
            else "../threaded_classification_experiments.py"
        ),
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


def test_get_classifier_by_name():
    """Test get_classifier_by_name method."""
    classifier_name_lists = [
        _get_classifier.convolution_based_classifiers,
        _get_classifier.deep_learning_classifiers,
        _get_classifier.dictionary_based_classifiers,
        _get_classifier.distance_based_classifiers,
        _get_classifier.feature_based_classifiers,
        _get_classifier.hybrid_classifiers,
        _get_classifier.interval_based_classifiers,
        _get_classifier.other_classifiers,
        _get_classifier.shapelet_based_classifiers,
        _get_classifier.vector_classifiers,
    ]

    # filled by _check_set_method
    classifier_list = []
    classifier_dict = {}
    all_classifier_names = []
    for classifier_name_list in classifier_name_lists:
        _check_set_method(
            get_classifier_by_name,
            classifier_name_list,
            classifier_list,
            classifier_dict,
            all_classifier_names,
        )

    _check_set_method_results(
        classifier_dict,
        estimator_name="Classifiers",
        method_name="get_classifier_by_name",
    )


def test_get_classifier_by_name_invalid():
    """Test get_classifier_by_name method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN CLASSIFIER"):
        get_classifier_by_name("invalid")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("EEGNet", "EEGNetClassifier"),
        ("DeepConvNet", "DeepConvNetClassifier"),
        ("RiemannianMDM", "RiemannianMDMClassifier"),
        ("RiemannianKNN", "RiemannianKNNClassifier"),
        ("CSPSVM", "CSPSVMClassifier"),
    ],
)
def test_get_aeon_neuro_classifier(name, expected):
    """Test native aeon-neuro classifiers are available by name and alias."""
    pytest.importorskip("aeon_neuro")
    for classifier_name in (name, expected, f"neuro:{expected}"):
        try:
            classifier = get_classifier_by_name(classifier_name)
        except ModuleNotFoundError as exc:
            pytest.skip(str(exc))
        assert type(classifier).__name__ == expected


def test_get_aeon_neuro_classifier_invalid():
    """Test that the prefixed lookup only accepts exported neuro classifiers."""
    pytest.importorskip("aeon_neuro")
    with pytest.raises(ValueError, match="UNKNOWN CLASSIFIER"):
        get_classifier_by_name("neuro:InvalidClassifier")


def test_get_future_aeon_neuro_classifier(monkeypatch):
    """Test discovery of a newly exported classifier and shared parameters."""
    classification = pytest.importorskip("aeon_neuro.classification")
    from aeon.classification.base import BaseClassifier

    class FutureClassifier(BaseClassifier):
        __module__ = "aeon_neuro.classification.future"

        def __init__(self, random_state=None, n_jobs=1):
            self.random_state = random_state
            self.n_jobs = n_jobs
            super().__init__()

        def _fit(self, X, y):
            return self

        def _predict(self, X):
            return [0] * len(X)

    monkeypatch.setattr(
        classification, "FutureClassifier", FutureClassifier, raising=False
    )
    classifier = get_classifier_by_name(
        "neuro:FutureClassifier", random_state=42, n_jobs=2
    )
    assert isinstance(classifier, FutureClassifier)
    assert (classifier.random_state, classifier.n_jobs) == (42, 2)


def test_aeon_classifiers_available():
    """Test all aeon classifiers are available."""
    excluded = [
        # composable/wrapper
        "ClassifierChannelEnsemble",
        "ClassifierPipeline",
        "ClassifierEnsemble",
        "SklearnClassifierWrapper",
        "IntervalForestClassifier",
        # ordinal
        "OrdinalTDE",
        "IndividualOrdinalTDE",
        # just missing
        "TDMVDCClassifier",
    ]

    est = [e for e, _ in all_estimators(type_filter="classifier")]
    for e in est:
        if e in excluded:
            continue

        try:
            assert get_classifier_by_name(e) is not None
        except ModuleNotFoundError:
            continue
