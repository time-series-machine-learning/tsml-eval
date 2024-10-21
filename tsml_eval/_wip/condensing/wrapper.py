import numpy as np
from aeon.classification.base import BaseClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.clustering import TimeSeriesKMeans


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
        metric="dtw",
        metric_params=None,
        classifier=None,
        num_instances_per_class=1,
    ):
        self.metric = metric

        self.metric_params = metric_params
        if self.metric_params is None:
            self.metric_params = {}

        self.num_instances_per_class = num_instances_per_class

        self.classifier = classifier
        if self.classifier is None:
            self.classifier = KNeighborsTimeSeriesClassifier(
                distance=self.metric,
                weights="distance",
                distance_params=self.metric_params,
                n_neighbors=1,
            )

        self.selected_series = []
        self.y_selected_series = []

        self.clusterer = TimeSeriesKMeans(
            n_clusters=self.num_instances_per_class,
            metric=self.metric,
            distance_params=self.metric_params,
            averaging_method="ba",
            average_params=self.metric_params,
        )

        super().__init__()

    def _fit(self, X, y):
        for i in np.unique(y):
            idxs_class = np.where(y == i)

            self.clusterer.fit(X[idxs_class])
            averaged_series_class_i = self.clusterer.cluster_centers_

            if len(averaged_series_class_i.shape) == 3:
                averaged_series_class_i = np.squeeze(averaged_series_class_i, axis=1)

            self.selected_series.append(averaged_series_class_i)
            self.y_selected_series.append(i)

        self.classifier.fit(
            np.array(self.selected_series), np.array(self.y_selected_series)
        )

        return self

    def _predict(self, X):
        return self.classifier.predict(X)
