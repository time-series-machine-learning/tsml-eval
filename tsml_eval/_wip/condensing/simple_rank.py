import numpy as np
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.distances import get_distance_function
from aeon.transformations.collection.base import BaseCollectionTransformer


class SimpleRankCondenser(BaseCollectionTransformer):
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
            self.metric_ = get_distance_function(metric=self.distance)

        self.selected_indices = []

        super().__init__()

    def _fit(self, X, y):
        # As SR do not separate prototypes per class, the number should be multiplied by
        # the number of instances per class of other methods.
        num_classes = len(np.unique(y))
        self.num_instances = self.num_instances * num_classes

    def _transform(self, X, y):
        n_samples = X.shape[0]
        rank = np.zeros(n_samples)
        distance = np.zeros(n_samples)
        num_classes = len(np.unique(y))

        for i in range(n_samples):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_pattern_loo = X[i]
            y_pattern_loo = y[i]

            # Consider moving this to the init method.
            classifier = KNeighborsTimeSeriesClassifier(
                distance=self.distance,
                distance_params=self.distance_params,
                n_neighbors=1,
            )

            classifier.fit(X_train, y_train)
            prediction = classifier.predict(X_pattern_loo)

            if y_pattern_loo == prediction:
                rank[i] = 1
            else:
                rank[i] = -2 / (num_classes - 1)

            # compute distance to nearest neighbour in class
            distance[i] = np.min(
                np.array(
                    [
                        self.metric_(
                            X_pattern_loo,
                            j,
                            **self.distance_params,
                        )
                        for j in X_train[np.where(y_train == y_pattern_loo)[0]]
                    ]
                )
            )
        order = sorted(zip(rank, -np.array(distance), range(n_samples)))[::-1]

        self.selected_indices = [x[2] for x in order][: self.num_instances]

        condensed_X, condensed_y = X[self.selected_indices], y[self.selected_indices]

        return condensed_X, condensed_y

    def _fit_transform(self, X, y):
        self._fit(X, y)
        return self._transform(X, y)
