"""Tests for classification estimator results."""

import numpy as np
import pytest

from tsml_eval.evaluation.storage.classifier_results import ClassifierResults
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


def test_classifier_results_with_one_present_class():
    """Test AUROC is undefined when only one training class is in the test set."""
    cr = ClassifierResults(
        n_classes=2,
        class_labels=np.array(["cat", "cat"]),
        predictions=np.array(["cat", "cat"]),
        probabilities=np.array([[0.9, 0.1], [0.8, 0.2]]),
        labels=["cat", "dog"],
    )

    cr.calculate_statistics()

    assert np.isnan(cr.auroc_score)


def test_classifier_results_loading_no_class_dict():
    """Test ClassifierResults loading and saving with no class dictionary."""
    cr = ClassifierResults().load_from_file(
        _TEST_RESULTS_PATH + "/classification/javaResultsFile.csv"
    )
    assert cr.labels is None
    assert cr.auroc_score == 1.0
