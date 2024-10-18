import numpy as np
from tsml_eval._wip.forecasting.base import BaseForecaster
from abc import ABC, abstractmethod
from typing import final
from sklearn.linear_model import LinearRegression
class BaseWindowForecaster(BaseForecaster):
    def __init__(self, window, horizon=1, regressor=LinearRegression()):
        self.regressor = regressor
        super().__init__(horizon, window)

    def fit(self, y, X=None):
        """Fit forecaster to time series.

        Split X into windows of length window and train the forecaster on each window
        to predict the horizon ahead.

        Parameters
        ----------
        X : Time series on which to learn a forecaster

        Returns
        -------
        self
            Fitted estimator
        """
        # Window data
        y2=np.lib.stride_tricks.sliding_window_view(y, window_shape=self.window)
        # Ignore the final horizon values: need to store these for pred with empty y
        y2=y2[:-self.horizon]
        # Extract y
        y=X[self.window+self.horizon-1:]
        self.regressor.fit(X2,y)
        return self

    def predict(self, y=None, X=None):
        """Predict values for time series X.

        NOTE: will not be able to predict the first
        """
        # TODO deal with case y =None
        return self.regressor.predict(y[-self.window:])

    def forecast(self, y, X=None):
        """Forecast values for time series X.

        NOTE: deal with horizons
        """
        self.fit(y,X)
        return self.predict(y[:self.window])
