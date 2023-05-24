# -*- coding: utf-8 -*-
import numpy as np
from aeon.distances import get_distance_function
from aeon.transformations.base import BaseTransformer

from tsml_eval.estimators.classification.distance_based import (
    KNeighborsTimeSeriesClassifier,
)


class Drop1(BaseTransformer):
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

        super(Drop1, self).__init__()

    def _fit(self, X, y):
        """
        Implement of the SimpleRank prototype selection approach.

        Parameters
        ----------
        X -- numpy array of shape (n_samples, n_features) representing the feature
        vectors  of the instances.
        y -- numpy array of shape (n_samples,) representing the corresponding class
        labels.
        k -- int, the desired number of prototypes to be selected.

        Returns
        -------
        self
        """
        n_samples = X.shape[0]

        associates = [[] for _ in range(n_samples)]
        kneighbors = [[] for _ in range(n_samples)]
        y_pred = []

        classifier = KNeighborsTimeSeriesClassifier(
            distance=self.distance,
            distance_params=self._distance_params,
            n_neighbors=self.n_neighbors + 1,
        )

        # Predicting class with the instance in the set.
        # Also getting the kneighbors and the associates of the instance.
        for i in range(n_samples):
            classifier.fit(X, y)
            y_pred.append(classifier.predict(X[i]))
            i_kneighbors, i_distances = classifier._kneighbors(X[i])

            i_kneighbors = [x[1] for x in sorted(zip(i_distances, i_kneighbors))]

            for j in i_kneighbors:
                associates[j].append(i)

            kneighbors[i] = i_kneighbors

        # Predicting class without the instance in the set.
        y_pred_wo_P = []
        for i in range(n_samples):
            X_wo_P = np.delete(X, i, axis=0)
            y_wo_P = np.delete(y, i)
            classifier.fit(X_wo_P, y_wo_P)
            y_pred_wo_P.append(classifier.predict(X[i]))

        X_S = X.copy()
        y_S = y.copy()

        for i in range(n_samples):
            # Num of associates correctly classified with i (or P) as neighbor.
            with_list = [
                j
                for j in associates[i]
                if ((i in kneighbors[j]) and (y[j] == y_pred[j]))
            ]

            # Num of associates correctly classified without i (or P) as neighbor.
            without_list = [j for j in associates[i] if (y[j] == y_pred_wo_P[j])]

            # Check if removing i (or P) is better.
            if len(without_list) >= len(with_list):
                # Remove P from S.
                i_S = self._find_index(i, X, X_S)
                X_S = np.delete(X_S, i_S, axis=0)
                y_S = np.delete(y_S, i_S)

                # Remove P from the kneighbors of the associates.
                for j in associates[i]:
                    kneighbors[j].remove(i)

                    # if self.n_neighbors + 1 >= len(X_S):
                    #     classifier = KNeighborsTimeSeriesClassifier(
                    #         distance=self.distance,
                    #         distance_params=self._distance_params,
                    #         n_neighbors=len(X_S),
                    #     )

                    # Find the next nearest neighbor for the j-th associate.
                    classifier.fit(X_S, y_S)
                    y_pred[j] = classifier.predict(X[j])
                    j_kneighbors, _ = classifier._kneighbors(X[j])
                    j_kneighbors = self._find_index(j_kneighbors, X_S, X)

                    j_neighbor = list(
                        set(j_kneighbors).symmetric_difference(set(kneighbors[j]))
                    )[0]

                    kneighbors[j].append(j_neighbor)
                    associates[j_neighbor].append(j)

                # Remove P from the associates of the neighbors.
                for j in kneighbors[i]:
                    associates[j].remove(i)

                associates[i] = []
                kneighbors[i] = []

            # The instance worth staying.
            else:
                self.selected_indices.append(i)
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

    def _find_index(self, values, training_set_instance, training_set_to_find):
        if isinstance(values, int):
            values = [values]

        index = [
            xdx
            for xdx, x in enumerate(training_set_to_find)
            for k in values
            if np.array_equal(x, training_set_instance[k])
        ]

        if len(index) == 1:
            return index[0]
        else:
            return index
