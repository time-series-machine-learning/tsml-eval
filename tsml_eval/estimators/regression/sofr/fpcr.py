# -*- coding: utf-8 -*-
"""FPCRegressor.

Classical Scalar on Function Regression approach that allows transforming
via B-spline if desired.
"""
from sktime.regression.base import BaseRegressor

from tsml_eval.estimators.regression.transformations import FPCATransformer

__author__ = ["David Guijo-Rubio"]
__all__ = ["FPCRegressor"]


class FPCRegressor(BaseRegressor):
    """Scalar on Function Regression using Functional Principal Component Analysis."""

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "python_dependencies": "scikit-fda",
    }

    def __init__(
        self,
        n_components=10,
        smooth=None,
        n_basis=None,
        order=None,
        save_transformed_data=False,
        regression_technique=None,
        n_jobs=1,
    ):
        self.n_components = n_components
        self.smooth = smooth
        self.n_basis = n_basis
        self.order = order
        self.save_transformed_data = save_transformed_data
        self.regression_technique = regression_technique
        self.n_jobs = n_jobs
        self.fpca = None

        if self.regression_technique is None:
            from sklearn.linear_model import LinearRegression

            from tsml_eval.estimators.regression.sklearn import SklearnToTsmlRegressor

            self.regression_technique = SklearnToTsmlRegressor(
                LinearRegression(fit_intercept=True, n_jobs=self.n_jobs)
            )

        super(FPCRegressor, self).__init__()

    def _fit(self, X, y):
        """Fit a pipeline on cases (X,y), where y is the target variable.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self.fpca = FPCATransformer(
            n_components=self.n_components,
            n_basis=self.n_basis,
            order=self.order,
            smooth=self.smooth,
        )

        X_t = self.fpca.fit_transform(X)

        self.regression_technique.fit(X_t, y)

        if self.save_transformed_data:
            self.transformed_data_ = X_t

        return self

    def _predict(self, X):
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        X_t = self.fpca.transform(X)
        return self.regression_technique.predict(X_t)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For regressors, a "default" set of parameters should be provided for
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
        pass
