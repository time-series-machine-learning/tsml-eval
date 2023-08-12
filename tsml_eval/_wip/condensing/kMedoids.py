import numpy as np
from aeon.clustering.k_medoids import TimeSeriesKMedoids
from aeon.transformations.collection.base import BaseCollectionTransformer


class kMedoidsCondenser(BaseCollectionTransformer):
    """
    Classifier wrapper for its use with any condensing approach.

    Parameters
    ----------
    distance
    distance_params

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
        num_instances_per_class=1,
        random_state=None,
    ):
        self.distance = distance

        self.distance_params = distance_params
        if self.distance_params is None:
            self.distance_params = {}

        self.num_instances_per_class = num_instances_per_class

        self.selected_series = []
        self.y_selected_series = []

        self.random_state = random_state

        super(kMedoidsCondenser, self).__init__()

    def _fit(self, X, y):
        self.num_instances_per_class = self.num_instances_per_class * len(np.unique(y))
        self.clusterer = TimeSeriesKMedoids(
            n_clusters=self.num_instances_per_class,
            distance=self.distance,
            distance_params=self.distance_params,
            method="pam",
            random_state=self.random_state,
        )

    def _transform(self, X, y):
        for i in np.unique(y):
            idxs_class = np.where(y == i)
            self.clusterer.fit(X[idxs_class])
            averaged_series_class_i = self.clusterer.cluster_centers_

            if len(averaged_series_class_i.shape) == 3:
                averaged_series_class_i = np.squeeze(averaged_series_class_i, axis=1)

            self.selected_series.append(averaged_series_class_i)
            self.y_selected_series.append(i)

        return np.array(self.selected_series), np.array(self.y_selected_series)

    def _fit_transform(self, X, y):
        self._fit(X, y)
        return self._transform(X, y)
