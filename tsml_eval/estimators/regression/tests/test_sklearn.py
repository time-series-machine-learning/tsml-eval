"""Tests for sklearn regressor wrapper."""

import numpy as np
import pytest
from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from sklearn.base import BaseEstimator, RegressorMixin

from tsml_eval.estimators.regression._sklearn import SklearnToAeonRegressor


class _SklearnMockRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.fit_shape_ = X.shape
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        self.predict_shape_ = X.shape
        return np.full(X.shape[0], self.mean_, dtype=self.mean_.dtype)


@pytest.mark.parametrize("as_list", [False, True])
def test_basic_univariate(as_list):
    """Wrap a standard sklearn-style regressor on univariate equal-length data."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=1, n_timepoints=4, regression_target=True, random_state=0
    )
    if as_list:
        X = [x.copy() for x in X]

    base = _SklearnMockRegressor()
    clf = SklearnToAeonRegressor(regressor=base)

    assert clf.get_tag("capability:multivariate") is False
    assert clf.get_tag("capability:unequal_length") is False

    clf.fit(X, y)
    pred = clf.predict(X)

    assert clf.regressor_ is not base
    assert not hasattr(base, "mean_")

    assert clf.regressor_.fit_shape_ == (6, 4)
    assert clf.regressor_.predict_shape_ == (6, 4)
    assert pred.shape == (6,)


@pytest.mark.parametrize("as_list", [False, True])
def test_concatenate_channels_flattens_multivariate(as_list):
    """Multivariate equal-length data should be flattened to 2D when requested."""
    X, y = make_example_3d_numpy(
        n_cases=5, n_channels=2, n_timepoints=4, regression_target=True, random_state=0
    )
    if as_list:
        X = [x.copy() for x in X]

    clf = SklearnToAeonRegressor(
        regressor=_SklearnMockRegressor(),
        concatenate_channels=True,
    )

    assert clf.get_tag("capability:multivariate") is True
    assert clf.get_tag("capability:unequal_length") is False

    clf.fit(X, y)
    pred = clf.predict(X)

    assert clf.regressor_.fit_shape_ == (5, 8)
    assert clf.regressor_.predict_shape_ == (5, 8)
    assert pred.shape == (5,)


def test_pad_unequal():
    """Unequal-length series should be padded to the train max length."""
    X_train = [
        np.array([[0.0, 1.0, 2.0]]),
        np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        np.array([[2.0, 3.0, 4.0, 5.0]]),
        np.array([[5.0, 4.0, 3.0]]),
    ]
    y_train = np.array([0, 1, 2, 3])

    X_test = [
        np.array([[0.0, 1.0]]),
        np.array([[1.0, 2.0, 3.0, 4.0]]),
    ]

    clf = SklearnToAeonRegressor(
        regressor=_SklearnMockRegressor(),
        pad_unequal=True,
    )

    assert clf.get_tag("capability:multivariate") is False
    assert clf.get_tag("capability:unequal_length") is True

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    assert clf.regressor_.fit_shape_ == (4, 5)
    assert clf.regressor_.predict_shape_ == (2, 5)
    assert pred.shape == (2,)


def test_rejects_multivariate_without_concatenation():
    """Multivariate data should be rejected when concatenate_channels is False."""
    X, y = make_example_3d_numpy(
        n_cases=5, n_channels=2, n_timepoints=4, regression_target=True, random_state=0
    )

    clf = SklearnToAeonRegressor(regressor=_SklearnMockRegressor())

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
        regression_target=True,
        random_state=0,
    )

    clf = SklearnToAeonRegressor(regressor=_SklearnMockRegressor())

    assert clf.get_tag("capability:unequal_length") is False

    with pytest.raises(ValueError, match="unequal length series"):
        clf.fit(X, y)
