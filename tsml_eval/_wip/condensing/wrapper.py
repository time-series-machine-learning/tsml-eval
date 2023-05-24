# -*- coding: utf-8 -*-
import numpy as np
from aeon.clustering.metrics.averaging._dba import dba
from aeon.distances import get_distance_function
from aeon.transformations.base import BaseTransformer


class WrapperBA(BaseTransformer):
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
        self._distance_params = distance_params
        if self._distance_params is None:
            self._distance_params = {}

        if isinstance(self.distance, str):
            self.metric_ = get_distance_function(metric=self.distance)

        self.selected_series = []
        self.y_selected_series = []

        super(WrapperBA, self).__init__()

    def _fit(self):
        """
        Implement the Wrapper for BA.

        Returns
        -------
        self
        """
        return self

    def _transform(self, X, y):
        for i in np.unique(y):
            idxs = np.where(y == i)

            self.selected_series.append(
                dba(X[idxs], metric=self.distance, kwargs=self._distance_params)
            )

            self.y_selected_series.append(i)
        return np.array(self.selected_series), np.array(self.y_selected_series)

    def _fit_transform(self, X, y):
        self._fit()
        condensed_X, condensed_y = self._transform(X, y)

        return condensed_X, condensed_y


# from aeon.datasets._single_problem_loaders import load_unit_test
# x_train, y_train = load_unit_test("TRAIN")

# wa = WrapperBA()
# x_condensed, y_condensed = wa._fit_transform(x_train, y_train)

# print(x_condensed)
# print(y_condensed)
