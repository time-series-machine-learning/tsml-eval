# -*- coding: utf-8 -*-
"""Basic knn regressor."""

__author__ = ["TonyBagnall", "GuiArcencio"]
__all__ = ["KNeighborsTimeSeriesRegressor"]

import numpy as np
from sktime.distances import distance_factory
from sktime.regression.base import BaseRegressor


class KNeighborsTimeSeriesRegressor(BaseRegressor):
    """Add docstring.

    Parameters
    ----------
    distance : str or Callable, default="euclidean"
        Metric to be used for distance computations.
    distance_params : dict or None, default=None
        Parameters used by the distance metric.
    n_neighbours : int, default=1
        Number of neighbours considered when predicting the target value.
    weights : {"uniform". "distance"}, default="uniform"
        Weight function used in prediction. "distance" means target value
        average is weighted by 1 / distance**2.
    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        distance="euclidean",
        distance_params=None,
        n_neighbours=1,
        weights="uniform",
    ):
        self.distance = distance
        self.distance_params = distance_params
        self.n_neighbours = n_neighbours
        self.weights = weights

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
        for i in range(X.shape[0]):
            distances = np.array(
                [self.metric(X[i], self._X[j]) for j in range(self._X.shape[0])]
            )

            # Find indices of k nearest neighbors using partitioning:
            # [0..k-1], [k], [k+1..n-1]
            # They might not be ordered within themselves,
            # but it is not necessary and partitioning is
            # O(n) while sorting is O(nlogn)
            closest_idx = np.argpartition(distances, self.n_neighbours)
            closest_idx = closest_idx[: self.n_neighbours]

            closest_targets = self._y[closest_idx]

            if self.weights == "distance":
                weight_vector = distances[closest_idx]
                weight_vector = weight_vector**2

                # Using epsilon ~= 0 to avoid division by zero
                weight_vector = 1 / (weight_vector + np.finfo(float).eps)
                preds[i] = np.average(closest_targets, weights=weight_vector)
            elif self.weights == "uniform":
                preds[i] = np.mean(closest_targets)
            else:
                raise Exception(f"Invalid kNN weights: {self.weights}")
        return preds
