"""Abstract base class"""

__maintainer__ = []
__all__ = ["BasePLA"]

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class BasePLA():
    """
        Base class for algorithms which use PLA (Piecewise Linear Approximation) for segmentation.

        Parameters
        ----------
        max_error: float
            The maximum error valuefor the function to find before segmenting the dataset
    """
    
    def __init__(self, max_error):
        self.max_error = max_error
    
    def linear_regression(self, time_series):
        """Returns the fitted line through a time series.

        Parameters
        ----------
        time_series : np.array
            1D time series.
      
        Returns
        -------
        np.array
            the fitted line
        """
        n = len(time_series)
        Y = np.array(time_series)
        X = np.arange(n).reshape(-1 , 1)
        linearRegression = LinearRegression()
        linearRegression.fit(X, Y)
        regression_line = np.array(linearRegression.predict(X))
        return regression_line
    
    def sum_squared_error(self, time_series, linear_regression_time_series):
        """Returns the squared sum error time series and its linear regression
        
        formula: sse = the sum of the differences of the original series
        against the predicted series squared

        Parameters
        ----------
        time_series : np.array
            1D time series.
        linear_regression_time_series: np.array
            1D linear time series formatted using linear regression

        Returns
        -------
        error
            the squared sum error of the split segmentations
        """
        
        error = np.sum((time_series - linear_regression_time_series) ** 2)
        return error
    
    def calculate_error(self, time_series):
        """Returns the squared sum error of a time series and its linear regression

        Parameters
        ----------
        time_series : np.array
            1D time series.

        Returns
        -------
        error
            the squared sum error of a time series and it's linear regression
        """
        
        lrts = self.linear_regression(time_series)
        sse = self.sum_squared_error(time_series, lrts)
        return sse
    
    def create_segment(self, time_series):
        """create a linear segment of a given time series.

        Parameters
        ----------
        time_series : np.array
            1D time series.

        Returns
        -------
        np.array
            the linear regression of the time series.
        """
        return self.linear_regression(time_series)