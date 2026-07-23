"""Tests for sklearn classifier wrapper."""

import numpy as np
import pytest
from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from sklearn.base import BaseEstimator, ClassifierMixin

from tsml_eval.estimators.classification._sklearn import SklearnToAeonClassifier


class _SklearnMockClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.fit_shape_ = X.shape
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        self.predict_shape_ = X.shape
        return np.full(X.shape[0], self.classes_[0], dtype=self.classes_.dtype)


class _SklearnProbaClassifier(_SklearnMockClassifier):
    def predict_proba(self, X):
        self.predict_proba_shape_ = X.shape
        proba = np.zeros((X.shape[0], len(self.classes_)))
        proba[:, 0] = 1.0
        return proba


@pytest.mark.parametrize("as_list", [False, True])
def test_basic_univariate(as_list):
    """Wrap a standard sklearn-style classifier on univariate equal-length data."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=1, n_timepoints=4, random_state=0
    )
    if as_list:
        X = [x.copy() for x in X]

    base = _SklearnProbaClassifier()
    clf = SklearnToAeonClassifier(classifier=base)

    assert clf.get_tag("capability:multivariate") is False
    assert clf.get_tag("capability:unequal_length") is False

    clf.fit(X, y)
    pred = clf.predict(X)
    proba = clf.predict_proba(X)

    assert clf.classifier_ is not base
    assert not hasattr(base, "classes_")

    assert clf.classifier_.fit_shape_ == (6, 4)
    assert clf.classifier_.predict_shape_ == (6, 4)
    assert clf.classifier_.predict_proba_shape_ == (6, 4)

    assert pred.shape == (6,)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("as_list", [False, True])
def test_concatenate_channels_flattens_multivariate(as_list):
    """Multivariate equal-length data should be flattened to 2D when requested."""
    X, y = make_example_3d_numpy(
        n_cases=5, n_channels=2, n_timepoints=4, random_state=0
    )
    if as_list:
        X = [x.copy() for x in X]

    clf = SklearnToAeonClassifier(
        classifier=_SklearnProbaClassifier(),
        concatenate_channels=True,
    )

    assert clf.get_tag("capability:multivariate") is True
    assert clf.get_tag("capability:unequal_length") is False

    clf.fit(X, y)
    pred = clf.predict(X)

    assert clf.classifier_.fit_shape_ == (5, 8)
    assert clf.classifier_.predict_shape_ == (5, 8)
    assert pred.shape == (5,)


def test_pad_unequal():
    """Unequal-length series should be padded to the train max length."""
    X_train = [
        np.array([[0.0, 1.0, 2.0]]),
        np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        np.array([[2.0, 3.0, 4.0, 5.0]]),
        np.array([[5.0, 4.0, 3.0]]),
    ]
    y_train = np.array([0, 1, 0, 1])

    X_test = [
        np.array([[0.0, 1.0]]),
        np.array([[1.0, 2.0, 3.0, 4.0]]),
    ]

    clf = SklearnToAeonClassifier(
        classifier=_SklearnProbaClassifier(),
        pad_unequal=True,
    )

    assert clf.get_tag("capability:multivariate") is False
    assert clf.get_tag("capability:unequal_length") is True

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    assert clf.classifier_.fit_shape_ == (4, 5)
    assert clf.classifier_.predict_shape_ == (2, 5)
    assert pred.shape == (2,)


def test_predict_proba_fall_back():
    """Classifiers without predict_proba should still support aeon predict_proba."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=1, n_timepoints=4, random_state=0
    )

    clf = SklearnToAeonClassifier(classifier=_SklearnMockClassifier())
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(proba)).issubset({0.0, 1.0})


def test_predict_proba_fall_back_multivariate():
    """Fallback predict_proba should convert multivariate data only once."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=4, random_state=0
    )

    clf = SklearnToAeonClassifier(
        classifier=_SklearnMockClassifier(),
        pad_unequal=True,
        concatenate_channels=True,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert clf.classifier_.predict_shape_ == (6, 8)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_rejects_multivariate_without_concatenation():
    """Multivariate data should be rejected when concatenate_channels is False."""
    X, y = make_example_3d_numpy(
        n_cases=5, n_channels=2, n_timepoints=4, random_state=0
    )

    clf = SklearnToAeonClassifier(classifier=_SklearnProbaClassifier())

    assert clf.get_tag("capability:multivariate") is False

    with pytest.raises(ValueError, match="multivariate series"):
        clf.fit(X, y)


def test_rejects_unequal_without_padding():
    """Unequal-length data should be rejected when pad_unequal is False."""
    X, y = make_example_3d_numpy_list(
        n_cases=10,
        n_channels=1,
        min_n_timepoints=4,
        max_n_timepoints=20,
        random_state=0,
    )

    clf = SklearnToAeonClassifier(classifier=_SklearnProbaClassifier())

    assert clf.get_tag("capability:unequal_length") is False

    with pytest.raises(ValueError, match="unequal length series"):
        clf.fit(X, y)
