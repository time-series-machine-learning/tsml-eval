import numpy as np
from tsml_eval._wip.forecasting.base import BaseForecaster
from abc import ABC, abstractmethod
from typing import final
from sklearn.linear_model import LinearRegression
class BaseWindowForecaster(BaseForecaster):
    def __init__(self, window, horizon=1, regressor=LinearRegression()):
        self.regressor = regressor
        super().__init__(horizon, window)

    def fit(self, X):
        """Fit forecaster to y, optionally using exogenous data X.

        Split y into windows of length window and train the forecaster on each window

        Parameters
        ----------
        X : Time series on which to learn a forecaster

        Returns
        -------
        self
            Fitted estimator
        """
        # Window data
        X2=np.lib.stride_tricks.sliding_window_view(X, window_shape=self.window)
        X2=X2[:-self.horizon]
        # Extract y
        y=X[self.window+self.horizon-1:]
        self.regressor.fit(X2,y)
        return self

    def predict(self, X):
        """Predict values for time series X.

        NOTE: will not be able to predict the first
        """
        X2 = np.lib.stride_tricks.sliding_window_view(X, window_shape=self.window)
        return self.regressor.predict(X2[self.horizon:])
