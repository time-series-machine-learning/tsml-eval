# -*- coding: utf-8 -*-
"""A standard sklearn regressor."""

__author__ = ["DGuijoRubio"]

import numpy as np
from aeon.regression.base import BaseRegressor


class SklearnBaseRegressor(BaseRegressor):
    """Wrapper for sklearn estimators."""

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(self, reg):
        self.reg = reg
        super(SklearnBaseRegressor, self).__init__()

    def _fit(self, X, y):
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

        self.reg.fit(X, y)

    def _predict(self, X) -> np.ndarray:
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

        return self.reg.predict(X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from sklearn.ensemble import RandomForestRegressor

        return {"reg": RandomForestRegressor(n_estimators=5)}
