# -*- coding: utf-8 -*-
"""A tsml wrapper for sklearn clusterers."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["SklearnToTsmlClusterer"]

import numpy as np
from sklearn.base import ClusterMixin
from tsml.base import BaseTimeSeriesEstimator, _clone_estimator


class SklearnToTsmlClusterer(ClusterMixin, BaseTimeSeriesEstimator):
    """Wrapper for sklearn estimators."""

    def __init__(self, clusterer, random_state=None):
        self.clusterer = clusterer
        self.random_state = random_state

        super(SklearnToTsmlClusterer, self).__init__()

    def fit(self, X, y):
        X = self._validate_data(X=X)

        if isinstance(X, np.ndarray):
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        else:
            raise ValueError("X must be a numpy array.")

        self._clusterer = _clone_estimator(self.clusterer, self.random_state)

        self._clusterer.fit(X, y)

    def predict(self, X) -> np.ndarray:
        X = self._validate_data(X=X, reset=False)

        if isinstance(X, np.ndarray):
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        else:
            raise ValueError("X must be a numpy array.")

        return self._clusterer.predict(X)
