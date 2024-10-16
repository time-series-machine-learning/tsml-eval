"""Base class for all forecasters.

basic model is to fix the horizon and possibly the window in constructor
model 1: y is the series to forecast, X is the exogenous data
pros: y standard name, predict looks standard
cons: order of y and X reversed, predict requires y, which looks weird

    def fit(self, y, X=None):
        y the series to train the forecaster, X possible exogenous data
    def predict(self, y, X=None):
        y the series to forecast, X possible exogenous data

"""

from aeon.base import BaseSeriesEstimator
from abc import ABC, abstractmethod


class BaseForecaster(BaseSeriesEstimator, ABC):
    """
    Abstract base class for time series forecasters.

    The base forecaster specifies the methods and method signatures that all
    forecasters have to implement. Attributes with an underscore suffix are set in the
    method fit.

    Attributes
    ----------
    horizon : int, default =1
        The number of time steps ahead to forecast. If horizon is one, the forecaster
        will learn to predict one point ahead
    window : int or None
        The window prior to the current time point to use in forecasting. So if
        horizon is one, forecaster will, train using points $i$ to $window+i-1$ to
        predict value $window+i$. If horizon is 4, forecaster will used points $i$
        to $window+i-1$ to predict value $window+i+3$. If None, the algorithm will
        internally determine what data to use to predict `horizon` steps ahead.

    To Do
    -----
    axis
    """

    def __init__(self, horizon=1, window=None):
        self.horizon = horizon
        self.window = window
        self._is_fitted = False
        super().__init__(axis=1)

    @abstractmethod
    def fit(self, X):
        """Fit forecaster to series X.

        Returns
        -------
        self
            Fitted estimator
        """
        ...

    @abstractmethod
    def predict(self, X):
        """

        Returns
        -------
        float
            single prediction.
        """
        ...
