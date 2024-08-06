"""Abstract base class for time series segmenters."""

__all__ = ["BaseForecaster"]
__maintainer__ = []

from abc import ABC, abstractmethod
from typing import List, final

import numpy as np
import pandas as pd

from aeon.base import BaseSeriesEstimator


class BaseForecaster(BaseSeriesEstimator, ABC):
    """Base class for forecasting algorithms.

    First mock up base class for forecasting. This makes some restrictive assumptions
    as to the use case. Series are stored in 1D or 2D numpy arrays. If 2D, the axis
    of time can be specified.

    Horizon is set in constructor. The horizon is the number of steps ahead to forecast.
    Fit constructs a model for predicting y, possibly using exongenoud data X. X and
    y are assumed to be aligned in time.

    Predict makes a single predition either based on input prediction series y or as
    the next expected value from fit.
    """

    _tags = {
        "X_inner_type": "np.ndarray",  # One of VALID_INNER_TYPES
    }

    def __init__(self, axis, horizon=1):
        self.horizon = horizon
        self._is_fitted = False
        super().__init__(axis=axis)

    @final
    def fit(self, y, X=None, axis=1):
        """Fit time series forecaster to y with possible exogenous data X.

        Returns
        -------
        self
            Fitted estimator
        """
        self._fit(X=X, y=y)
        self._is_fitted = True
        return self

    @final
    def predict(self, y=None, X=None, axis=1):
        """Make a predition.

        Returns
        -------
        List
            Either a list of indexes of X indicating where each segment begins or a
            list of integers of ``len(X)`` indicating which segment each time point
            belongs to.
        """
        self.check_is_fitted()
        if axis is None:
            axis = self.axis
        return self._predict(y, X)

    def fit_predict(self, X, y=None, axis=1):
        """Fit segmentation to data and return it."""
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        self.fit(X, y, axis=axis)
        return self.predict(X, axis=axis)

    def _fit(self, X, y):
        """Fit time series classifier to training data."""
        return self

    @abstractmethod
    def _predict(self, y, X) -> np.ndarray:
        """Create and return a segmentation of X."""
        ...
