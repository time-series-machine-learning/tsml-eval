# -*- coding: utf-8 -*-
"""Mean label regressor.

dummy approach to serve as deadline.
"""

__author__ = ["David Guijo-Rubio"]
__all__ = ["MeanPredictorRegressor"]


import numpy as np
from aeon.regression.base import BaseRegressor


class MeanPredictorRegressor(BaseRegressor):
    """Dummy regressor that estimates mean for all testing time series."""

    _tags = {"capability:multivariate": True}

    def __init__(self):
        self.output_mean = 0
        super(MeanPredictorRegressor, self).__init__()

    def _fit(self, X, y):
        self.output_mean = np.mean(y)

    def _predict(self, X) -> np.ndarray:
        n_instances, _, _ = X.shape
        output = [self.output_mean] * n_instances
        return output
