# -*- coding: utf-8 -*-
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
    ):
        self.distance = distance

        self.distance_params = distance_params
        if self.distance_params is None:
            self.distance_params = {}

        self.num_instances_per_class = num_instances_per_class

        self.selected_series = []
        self.y_selected_series = []

        self.clusterer = TimeSeriesKMeans(
            n_clusters=self.num_instances_per_class,
            metric=self.distance,
            distance_params=self.distance_params,
            averaging_method="ba",
            average_params=self.distance_params,
        )

        super(kMeansCondenser, self).__init__()

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
        return self._transform(X, y)
