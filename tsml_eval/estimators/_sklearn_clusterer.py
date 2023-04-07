# -*- coding: utf-8 -*-
"""A tsml wrapper for sklearn clusterers."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["SklearnToTsmlClusterer"]

import numpy as np
from sklearn.base import ClusterMixin
from sklearn.utils.validation import check_is_fitted
from tsml.base import BaseTimeSeriesEstimator, _clone_estimator


class SklearnToTsmlClusterer(ClusterMixin, BaseTimeSeriesEstimator):
    """Wrapper for sklearn estimators."""

    def __init__(self, clusterer=None, concatenate_channels=False, random_state=None):
        self.clusterer = clusterer
        self.concatenate_channels = concatenate_channels
        self.random_state = random_state

        super(SklearnToTsmlClusterer, self).__init__()

    def fit(self, X, y=None):
        if self.clusterer is None:
            raise ValueError("Clusterer not set")

        X = self._validate_data(X=X)
        X = self._convert_X(X, concatenate_channels=self.concatenate_channels)

        self._clusterer = _clone_estimator(self.clusterer, self.random_state)
        self._clusterer.fit(X, y)

        self.labels_ = self._clusterer.labels_

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X, concatenate_channels=self.concatenate_channels)

        return self._clusterer.predict(X)

    def _more_tags(self):
        return {"X_types": ["2darray"]}
