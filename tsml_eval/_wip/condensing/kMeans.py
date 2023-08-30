import numpy as np
from aeon.clustering.k_means import TimeSeriesKMeans
from aeon.transformations.collection.base import BaseCollectionTransformer


class kMeansCondenser(BaseCollectionTransformer):
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
        "fit_is_empty": True,
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

        self.selected_series = np.array([])
        self.y_selected_series = []

        self.random_state = random_state

        self.clusterer = TimeSeriesKMeans(
            n_clusters=self.num_instances_per_class,
            metric=self.distance,
            distance_params=self.distance_params,
            averaging_method="ba",
            average_params=self.distance_params,
            random_state=self.random_state,
        )

        super(kMeansCondenser, self).__init__()

    def _transform(self, X, y):
        self.selected_series = self.selected_series.reshape(0, *X.shape[1:])

        for i in np.unique(y):
            idxs_class = np.where(y == i)

            self.clusterer.fit(X[idxs_class])
            averaged_series_class_i = self.clusterer.cluster_centers_

            self.selected_series = np.concatenate(
                (self.selected_series, averaged_series_class_i), axis=0
            )

            self.y_selected_series.extend([i] * self.num_instances_per_class)

        return np.array(self.selected_series), np.array(self.y_selected_series)

    def _fit_transform(self, X, y):
        return self._transform(X, y)
