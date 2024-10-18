"""Base class for all forecasters.

basic model is to fix the horizon and possibly the window in constructor
model 1: y is the series to forecast, X is the exogenous data

pros: y standard name, predict looks standard
cons: order of y and X reversed, predict requires y, which looks weird

ignore exogenous for now.

    def fit(self, X):
        y the series to train the forecaster
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

    Parameters
    ----------
    horizon : int, default =1
        The number of time steps ahead to forecast. If horizon is one, the forecaster
        will learn to predict one point ahead
    window : int or None
        The window prior to the current time point to use in forecasting. So if
        horizon is one, forecaster will train using points $i$ to $window+i-1$ to
        predict value $window+i$. If horizon is 4, forecaster will used points $i$
        to $window+i-1$ to predict value $window+i+3$. If None, the algorithm will
        internally determine what data to use to predict `horizon` steps ahead.
    """

    # TODO: add any forecasting specific tags
    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "X_inner_type": "np.ndarray",  # one of VALID_INNER_TYPES
    }
    def __init__(self, horizon=1, window=None, axis=1):
        self.horizon = horizon
        self.window = window
        self._is_fitted = False
        super().__init__(axis)

    @abstractmethod
    def fit(self, y, X=None):
        """Fit forecaster to series X.
        TODO: passing series as X makes sense in machine learning, but not in
        forecasting

        Parameters
        -------
        y : np.ndarray
            A time series on which to learn a forecaster to predict horizon ahead
        X : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        self
            Fitted BaseForecaster.
        """
        ...

    @abstractmethod
    def predict(self, y=None, X=None):
        """

        Parameters
        -------
        Parameters
        -------
        y : np.ndarray, default = None
            A time series to predict the next horizon value for. If None,
            predict the next horizon value after series seen in fit.
        X : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        float
            single prediction.
        """
        ...

    @abstractmethod
    def forecast(self, y, X=None):
        """

        basically fit_predict.

        Returns
        -------
        np.ndarray
            single prediction directly after the last point in X.
        """
        ...
