# -*- coding: utf-8 -*-
import numpy as np
from aeon.distances import get_distance_function
from aeon.transformations.base import BaseTransformer

from tsml_eval.estimators.classification.distance_based import (
    KNeighborsTimeSeriesClassifier,
)


class SimpleRank(BaseTransformer):
    """
    Class for the simple_rank condensing approach.

    Parameters
    ----------
    distance
    distance_params
    n_neighbors

    References
    ----------
    .. [1] Ueno, K., Xi, X., Keogh, E., & Lee, D. J. (2006, December). Anytime
    classification using the nearest neighbor algorithm with applications to stream
    mining. In Sixth International Conference on Data Mining (ICDM'06) (pp. 623-632).
    IEEE.

    Examples
    --------
     >>> from ...
     >>> from ...
    """

    _tags = {
        "univariate-only": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        distance="dtw",
        distance_params=None,
        n_neighbors=1,
    ):
        self.distance = distance
        self._distance_params = distance_params
        if self._distance_params is None:
            self._distance_params = {}

        self.n_neighbors = n_neighbors

        if isinstance(self.distance, str):
            self.metric_ = get_distance_function(metric=self.distance)

        self.selected_indices = []

        super(SimpleRank, self).__init__()

    def _fit(self, X, y):
        """
        Implement of the SimpleRank prototype selection approach.

        Parameters
        ----------
        X -- numpy array of shape (n_samples, n_features) representing the feature
        vectors  of the instances.
        y -- numpy array of shape (n_samples,) representing the corresponding class
        labels.

        Returns
        -------
        self
        """
        n_samples = X.shape[0]
        rank = np.zeros(n_samples)
        distance = np.zeros(n_samples)
        num_classes = len(np.unique(y))

        for i in range(n_samples):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_pattern_loo = X[i]

            classifier = KNeighborsTimeSeriesClassifier(
                distance=self.distance,
                distance_params=self._distance_params,
                n_neighbors=self.n_neighbors,
            )

            classifier.fit(X_train, y_train)
            prediction = classifier.predict(X_pattern_loo)

            if y[i] == prediction:
                rank[i] = 1
            else:
                rank[i] = -2 / (num_classes - 1)

            # compute distance to nearest neigh in class
            distance[i] = np.min(
                np.array(
                    [
                        self.metric_(
                            X_pattern_loo,
                            X_train[np.where(y_train == y[i])[0]][j],
                            **self._distance_params,
                        )
                        for j in range(len([np.where(y_train == y[i])[0]]))
                    ]
                )
            )

        samples_ordered = sorted(zip(rank, -np.array(distance), range(n_samples)))

        self.selected_indices = [x[2] for x in samples_ordered][::-1][
            : self.n_neighbors
        ]

        return self

    def _transform(self, X, y):
        return X[self.selected_indices], y[self.selected_indices]

    def _fit_transform(self, X, y):
        self._fit(X, y)
        condensed_X, condensed_y = self._transform(X, y)

        return condensed_X, condensed_y

    def _get_selected_indices(self):
        # todo: check that fit has already been called.
        return self.selected_indices
