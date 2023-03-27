# -*- coding: utf-8 -*-
import numpy as np
import pyfftw
from sklearn.ensemble import ExtraTreesClassifier
from sktime.classification import BaseClassifier
from statsmodels.regression.linear_model import burg

from tsml_eval.estimators.classification.transformations.supervised_intervals import (
    SupervisedIntervals,
)


class RSTSF(BaseClassifier):
    def __init__(
        self,
        n_estimators=500,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state

        super(RSTSF, self).__init__()

    def _fit(self, X, y):
        transforms = self._transform_data(X.reshape((X.shape[0], -1)))
        Xt = np.empty((X.shape[0], 0))
        self.transformers = []
        for i, t in enumerate(transforms):
            t = t.reshape((t.shape[0], 1, t.shape[1]))
            si = SupervisedIntervals(random_state=self.random_state)
            Xt = np.hstack((Xt, si.fit_transform(t, y)))
            self.transformers.append(si)

        self.clf = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion="entropy",
            class_weight="balanced",
            max_features="sqrt",
        )
        self.clf.fit(Xt, y)

        return self

    def _predict(self, X):
        transforms = self._transform_data(X.reshape((X.shape[0], -1)))
        Xt = np.empty((X.shape[0], 0))
        for i, t in enumerate(transforms):
            t = t.reshape((t.shape[0], 1, t.shape[1]))
            si = self.transformers[i]
            Xt = np.hstack((Xt, si.transform(t)))

        return self.clf.predict(Xt)

    def _predict_proba(self, X):
        transforms = self._transform_data(X.reshape((X.shape[0], -1)))
        Xt = np.empty((X.shape[0], 0))
        for i, t in enumerate(transforms):
            t = t.reshape((t.shape[0], 1, t.shape[1]))
            si = self.transformers[i]
            Xt = np.hstack((Xt, si.transform(t)))

        return self.clf.predict_proba(Xt)

    def _transform_data(self, X):
        per_X = _getPeriodogramRepr(X)
        diff_X = np.diff(X)
        ar_X = _ar_coefs(X)
        ar_X[np.isnan(ar_X)] = 0

        return [X, per_X, diff_X, ar_X]

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
        return {
            "n_estimators": 2,
        }


def _getPeriodogramRepr(X):
    nfeats = X.shape[1]
    fft_object = pyfftw.builders.fft(X)
    per_X = np.abs(fft_object())
    return per_X[:, : int(nfeats / 2)]


def _ar_coefs(X):
    X_transform = []
    lags = int(12 * (X.shape[1] / 100.0) ** (1 / 4.0))
    for i in range(X.shape[0]):
        coefs, _ = burg(X[i, :], order=lags)
        X_transform.append(coefs)
    return np.array(X_transform)
