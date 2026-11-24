"""Tests for classification estimator results."""

import os

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from tsml_eval.evaluation.storage.classifier_results import (
    ClassifierResults,
    _calculate_missing_class_auroc_score,
)
from tsml_eval.experiments import run_classification_experiment
from tsml_eval.experiments.tests import _CLASSIFIER_RESULTS_PATH
from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH


def test_classifier_results_with_class_missing_from_test_set():
    """Test loading classifier results when a training class is absent in test."""
    cr = ClassifierResults().load_from_file(
        _TEST_RESULTS_PATH + "/broken/missingClassResultsFile.csv"
    )

    assert cr.labels == [0, 1, 2, 3]
    assert cr.probabilities.shape == (14, 4)
    assert cr.log_loss == pytest.approx(0.6040039230162959)
    assert cr.auroc_score == pytest.approx(0.9285714285714286)


@pytest.mark.parametrize(
    "y_train",
    [
        np.array(["a", "b", "c"] * 4),
        np.array([-1, 1, 2] * 4),
        np.array([1.5, 2.5, 3.5] * 4),
    ],
    ids=["str", "int", "float"],
)
def test_experiment_results_with_class_missing_from_test_set(y_train):
    """Test labels are loaded from the encoder dictionary written by experiments."""
    rng = np.random.RandomState(0)
    y_test = np.array([y_train[0], y_train[1]] * 3)
    results_path = _CLASSIFIER_RESULTS_PATH
    dataset_name = f"MissingClass{y_train.dtype.kind}"

    run_classification_experiment(
        rng.random_sample((len(y_train), 1, 10)),
        y_train,
        rng.random_sample((len(y_test), 1, 10)),
        y_test,
        DummyClassifier(),
        results_path,
        classifier_name="DummyClassifier",
        dataset_name=dataset_name,
        resample_id=0,
        benchmark_time=False,
    )

    test_file = (
        f"{results_path}DummyClassifier/Predictions/{dataset_name}/testResample0.csv"
    )
    cr = ClassifierResults().load_from_file(test_file)
    os.remove(test_file)

    assert cr.labels == [0, 1, 2]
    assert cr.probabilities.shape == (6, 3)
    assert np.isfinite(cr.log_loss)
    assert np.isfinite(cr.auroc_score)


def test_classifier_results_with_class_missing_and_no_class_dict():
    """Test loading fails when a class is missing and there is no class dictionary."""
    with pytest.raises(ValueError, match="different number of classes"):
        ClassifierResults().load_from_file(
            _TEST_RESULTS_PATH + "/broken/missingClassNoDictResultsFile.csv"
        )


def test_classifier_results_with_non_numeric_labels():
    """Test calculating statistics with a list of string class labels."""
    cr = ClassifierResults(
        n_classes=2,
        class_labels=np.array(["cat", "dog", "cat", "dog"]),
        predictions=np.array(["cat", "dog", "cat", "dog"]),
        probabilities=np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]),
        labels=["cat", "dog"],
    )

    cr.calculate_statistics()

    assert cr.labels == ["cat", "dog"]
    assert cr.log_loss == pytest.approx(0.164252033486018)
    assert cr.auroc_score == 1.0


def test_classifier_results_with_array_labels():
    """Test calculating statistics with class labels as a numpy array."""
    cr = ClassifierResults(
        n_classes=3,
        class_labels=np.array([0, 1, 0, 1]),
        predictions=np.array([0, 1, 0, 1]),
        probabilities=np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1]]
        ),
        labels=np.array([0, 1, 2]),
    )

    cr.calculate_statistics()

    assert np.isfinite(cr.log_loss)
    assert cr.auroc_score == 1.0


@pytest.mark.parametrize(
    "class_labels, probabilities, labels, match",
    [
        ([0, 1, 0], [[0.5, 0.5, 0.5]] * 3, [0, 1, 2], "should sum up to 1.0"),
        ([0, 1, 0], [[0.8, 0.1, 0.1]] * 3, [0, 1, 1], "must be unique"),
        ([0, 1, 0], [[0.8, 0.1, 0.1]] * 3, [1, 0, 2], "must be ordered"),
        ([0, 1, 0], [[0.8, 0.1, 0.1]] * 3, [0, 1], "Number of given labels"),
        ([0, 5, 0], [[0.8, 0.1, 0.1]] * 3, [0, 1, 2], "contains labels not in"),
    ],
    ids=["probabilities", "unique", "ordered", "n_labels", "unknown_label"],
)
def test_missing_class_auroc_score_invalid_input(
    class_labels, probabilities, labels, match
):
    """Test missing class AUROC raises the roc_auc_score input validation errors."""
    with pytest.raises(ValueError, match=match):
        _calculate_missing_class_auroc_score(
            np.array(class_labels), np.array(probabilities), labels
        )


def test_classifier_results_with_one_present_class():
    """Test AUROC errors when only one training class is in the test set."""
    cr = ClassifierResults(
        n_classes=2,
        class_labels=np.array(["cat", "cat"]),
        predictions=np.array(["cat", "cat"]),
        probabilities=np.array([[0.9, 0.1], [0.8, 0.2]]),
        labels=["cat", "dog"],
    )

    with pytest.raises(ValueError, match="fewer than two classes"):
        cr.calculate_statistics()


def test_classifier_results_loading_no_class_dict():
    """Test ClassifierResults loading with no class dictionary."""
    cr = ClassifierResults().load_from_file(
        _TEST_RESULTS_PATH + "/classification/javaResultsFile.csv"
    )
    assert cr.labels is None
    assert cr.auroc_score == 1.0
