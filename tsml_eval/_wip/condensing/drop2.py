# -*- coding: utf-8 -*-
import numpy as np
from aeon.distances import get_distance_function
from aeon.transformations.collection.base import BaseCollectionTransformer


class Drop2Condenser(BaseCollectionTransformer):
    """
    Class for the simple_rank condensing approach.

    Parameters
    ----------
    distance
    distance_params
    num_instances_per_class

    References
    ----------
    .. [1] Wilson, D. R., & Martinez, T. R. (2000). Reduction techniques for
    instance-based learning algorithms. Machine learning, 38, 257-286.

    Examples
    --------
     >>> from ...
     >>> from ...
    """

    _tags = {
        "univariate-only": True,
        "fit_is_empty": False,
        "X_inner_mtype": ["np-list", "numpy3D"],
        "requires_y": True,
        "y_inner_mtype": ["numpy1D"],
    }

    def __init__(
        self,
        distance="dtw",
        distance_params=None,
        num_instances=1,
    ):
        self.distance = distance
        self.distance_params = distance_params
        if self.distance_params is None:
            self.distance_params = {}

        self.num_instances = num_instances

        if isinstance(self.distance, str):
            self.metric = get_distance_function(metric=self.distance)

        self.selected_indices = []

        super(Drop2Condenser, self).__init__()

    def _fit(self, X, y):
        n_classes = len(np.unique(y))
        self.num_instances = self.num_instances * n_classes

    def _transform(self, X, y):
        """
        Implement of the Drop1 prototype selection approach.

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
        weights = [[] for _ in range(n_samples)]
        distance_nearest_enemy = []
        distances = np.zeros((n_samples, n_samples))

        # Getting the kneighbors and the associates of the instance.
        for p in range(n_samples):
            for p2 in range(p + 1, n_samples):
                distances[p, p2] = self.metric(X[p], X[p2], **self.distance_params)
                distances[p2, p] = distances[p, p2]

        for p in range(n_samples):
            weights[p], kneighbors[p], y_ordered = zip(
                *sorted(zip(distances[p], range(n_samples), y))
            )

            # todo: maybe removing first element as is itself?
            weights[p], kneighbors[p] = weights[p][1:], kneighbors[p][1:]

            for j in kneighbors[p][: self.num_instances]:
                associates[j].append(p)

            # Drop2 order instances by their distance to the nearest enemy.
            for kdx, _ in enumerate(kneighbors[p]):
                if y_ordered[kdx] != y[p]:
                    distance_nearest_enemy.append(weights[p][kdx])
                    break

        _, n_samples_ordered = zip(
            *sorted(zip(distance_nearest_enemy, range(n_samples)))
        )

        # Predicting with/without rule for each instance p in the set.
        for p in n_samples_ordered:
            without_P = 0
            with_P = 0

            for a in associates[p]:
                # WITH
                y_pred_w_P = self._predict_KNN(
                    kneighbors[a],
                    weights[a],
                    y,
                    self.num_instances,
                )

                if y_pred_w_P == y[a]:
                    with_P += 1
                # WITHOUT
                y_pred_wo_P = self._predict_KNN(
                    [k for k in kneighbors[a] if k != p],
                    [w for idx, w in enumerate(weights[a]) if idx != p],
                    y,
                    self.num_instances,
                )

                if y_pred_wo_P == y[a]:
                    without_P += 1

            if without_P < with_P:  # the instance is worth keeping.
                print(f"Keeping instance {p}.")
                self.selected_indices.append(p)
            else:  # the instance is not worth keeping.
                print(f"Removing instance {p}.")
                for a in associates[p]:
                    kneighbors[a] = [kn for kn in kneighbors[a] if kn != p]
                    for j in kneighbors[a][: self.num_instances]:
                        if a not in associates[j]:
                            associates[j].append(a)

        print(self.selected_indices)
        return X[self.selected_indices], y[self.selected_indices]

    def _fit_transform(self, X, y):
        self.fit(X, y)
        return self._transform(X, y)

    def _predict_KNN(self, neighbors, weights, y, num_neighbors):
        neighbors = neighbors[:(num_neighbors)]
        weights = weights[:(num_neighbors)]
        classes_, y_ = np.unique(y, return_inverse=True)
        scores = np.zeros(len(classes_))
        for id, w in zip(neighbors, weights):
            predicted_class = y_[id]
            scores[predicted_class] += 1 / (w + np.finfo(float).eps)
        return classes_[np.argmax(scores)]
