# -*- coding: utf-8 -*-
import numpy as np
from aeon.classification.base import BaseClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.clustering.metrics.averaging import elastic_barycenter_average


class WrapperBA(BaseClassifier):
    """
    Wrapper for BA methods using condensing approach.

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
    }

    def __init__(
        self,
        distance="dtw",
        distance_params=None,
    ):
        self.distance = distance
        self.distance_params = distance_params
        if self.distance_params is None:
            self.distance_params = {}

        self.selected_series = []
        self.y_selected_series = []

        self.classifier = KNeighborsTimeSeriesClassifier(
            distance=self.distance, distance_params=self.distance_params
        )

        super(WrapperBA, self).__init__()

    def _fit(self, X, y):
        for i in np.unique(y):
            idxs = np.where(y == i)

            series = elastic_barycenter_average(
                X[idxs],
                metric=self.distance,
                **self.distance_params,
            )

            if len(series.shape) == 3:
                series = np.squeeze(series, axis=0)

            self.selected_series.append(series)

            self.y_selected_series.append(i)

        self.classifier.fit(
            np.array(self.selected_series), np.array(self.y_selected_series)
        )
        return self

    def _predict(self, X):
        return self.classifier.predict(X)
