"""Abstract base class"""

__maintainer__ = []
__all__ = ["BasePLA"]

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class BasePLA():
    "Base class for piecewise linear approximation (PLA)"
    
    def __init__(self, max_error):
        self.max_error = max_error
    
    def linear_regression(self, time_series, sequence = None):
        n = len(time_series)
        Y = np.array(time_series)
        X = np.arange(n).reshape(-1 , 1)
        linearRegression = LinearRegression()
        linearRegression.fit(X, Y)
        regression_line = np.array(linearRegression.predict(X))
        return regression_line
    
    def sum_squared_error(self, time_series, linear_regression_time_series):
        "formula: sse = the sum of the differences of the original series against the predicted series squared"
        error = np.sum((time_series - linear_regression_time_series) ** 2)
        return error
    
    def calculate_error(self, time_series):
        lrts = self.linear_regression(time_series)
        sse = self.sum_squared_error(time_series, lrts)
        return sse
    
    def create_segment(self, time_series):
        return self.linear_regression(time_series)