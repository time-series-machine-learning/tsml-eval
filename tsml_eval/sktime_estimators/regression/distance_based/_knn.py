# -*- coding: utf-8 -*-
"""Basic knn regressor."""

__author__ = ["TonyBagnall"]
__all__ = ["KNeighborsTimeSeriesRegressor"]

import numpy as np
from sktime.distances import distance_factory
from sktime.regression.base import BaseRegressor


class KNeighborsTimeSeriesRegressor(BaseRegressor):
    """Add docstring."""

    _tags = {
        "X_inner_mtype": "numpy3d",
        "capability:multivariate": True,
    }

    def __init__(self, distance="euclidean", distance_params=None, n_neighbours=1):
        self.distance = distance
        self.distance_params = distance_params
        self.n_neighbours = n_neighbours
        super(KNeighborsTimeSeriesRegressor, self).__init__()

    def _fit(self, X, y) -> np.ndarray:
        """Fit the dummy regressor.

        Parameters
        ----------
        X : numpy ndarray with shape(n,d,m)
        y : numpy ndarray shape = [n_instances] - the target values

        Returns
        -------
        self : reference to self.
        """
        if isinstance(self.distance, str):
            if self.distance_params is None:
                self.metric = distance_factory(X[0], X[0], metric=self.distance)
            else:
                self.metric = distance_factory(
                    X[0], X[0], metric=self.distance, **self.distance_params
                )

        self._X = X
        self._y = y
        return self

    def _predict(self, X) -> np.ndarray:
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : numpy array of shape, shape (n, d, m)

        Returns
        -------
        y : predictions of target values for X, np.ndarray
        """
        # Measure distance between train set (self_X) and test set (X)
        preds = np.zeros(X.shape[0])
        distances = np.zeros(self._X.shape[0])
        # All one with aligned lists for prototype, very inefficient no doubt but
        # memory light
        for i in range(0, X.shape[0]):
            index = list(range(0, self._X.shape[0]))
            for j in range(0, self._X.shape[0]):
                distances[j] = self.metric(X[i], self._X[j])
            # Sort distances ascending. Could be done more efficiently
            distances, index = map(list, zip(*sorted(zip(distances, index))))
            # average top k values
            preds[i] = distances[0]
            for k in range(1, self.n_neighbours):
                preds[i] = preds[i] + distances[k]
            preds[i] = preds[i] / self.n_neighbours
        return preds
