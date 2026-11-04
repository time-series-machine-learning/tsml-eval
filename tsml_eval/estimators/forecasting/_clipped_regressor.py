"""Regressor wrapper that clips predictions to the range of its training targets."""

__all__ = ["ClippedRegressor"]

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class ClippedRegressor(RegressorMixin, BaseEstimator):
    """Clip a regressor's predictions to ``[min(y), max(y)]`` seen in ``fit``.

    Tree models never predict outside their training targets, but linear models can.
    When a forecaster feeds its own predictions back as inputs (iterative/recursive
    forecasting), an extrapolating linear model can compound into an exponentially
    exploding forecast, e.g. ``ESNormalisedRegressionForecaster`` with ``log=True``
    and a ridge regressor on a handful of M4 series. Clipping each one-step prediction
    keeps the recursion bounded.

    Parameters
    ----------
    regressor : sklearn regressor
        The regressor to wrap.
    """

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        y = np.asarray(y)
        self.regressor_ = clone(self.regressor).fit(X, y)
        self.min_, self.max_ = float(np.min(y)), float(np.max(y))
        return self

    def predict(self, X):
        return np.clip(self.regressor_.predict(X), self.min_, self.max_)
