# -*- coding: utf-8 -*-
"""A tsml wrapper for sklearn regressors."""

__author__ = ["DGuijoRubio", "MatthewMiddlehurst"]
__all__ = ["SklearnToTsmlRegressor"]

import numpy as np
from sklearn.base import ClassifierMixin
from tsml.base import BaseTimeSeriesEstimator, _clone_estimator


class SklearnToTsmlRegressor(ClassifierMixin, BaseTimeSeriesEstimator):
    """Wrapper for sklearn estimators."""

    def __init__(self, regressor, random_state=None):
        self.regressor = regressor
        self.random_state = random_state

        super(SklearnToTsmlRegressor, self).__init__()

    def fit(self, X, y):
        X, y = self._validate_data(X=X, y=y)

        if isinstance(X, np.ndarray):
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        else:
            raise ValueError("X must be a numpy array.")

        self._regressor = _clone_estimator(self.regressor, self.random_state)

        self._regressor.fit(X, y)

    def predict(self, X) -> np.ndarray:
        X = self._validate_data(X=X, reset=False)

        if isinstance(X, np.ndarray):
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        else:
            raise ValueError("X must be a numpy array.")

        return self._regressor.predict(X)
