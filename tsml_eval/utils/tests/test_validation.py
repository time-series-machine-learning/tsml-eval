"""Tests for validation utilities."""

from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from tsml.interval_based import DrCIFClassifier

from tsml_eval.utils.validation import (
    is_sklearn_classifier,
    is_sklearn_clusterer,
    is_sklearn_estimator,
    is_sklearn_regressor,
)


def test_is_sklearn_estimator():
    """Test is_sklearn_estimator."""
    assert is_sklearn_estimator(DummyClassifier())
    assert not is_sklearn_estimator(DrCIFClassifier())


def test_is_sklearn_classifier():
    """Test is_sklearn_classifier."""
    assert is_sklearn_classifier(DummyClassifier())
    assert not is_sklearn_classifier(DummyRegressor())


def test_is_sklearn_regressor():
    """Test is_sklearn_regressor."""
    assert is_sklearn_regressor(DummyRegressor())
    assert not is_sklearn_regressor(DummyClassifier())


def test_is_sklearn_clusterer():
    """Test is_sklearn_clusterer."""
    assert is_sklearn_clusterer(KMeans())
    assert not is_sklearn_clusterer(DummyClassifier())
