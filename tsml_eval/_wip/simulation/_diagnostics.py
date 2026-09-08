"""Diagnostics for the explanatory simulation protocol.

These are not competitors. They are reference points that answer questions about
the generated problem rather than about any method's design, and they are
excluded from method-family rankings.

``GlobalSummaryDiagnostic`` computes a handful of statistics over the whole
series and classifies on those alone. It has no notion of an interval, so it
cannot localise anything. Its accuracy therefore measures how much of a
condition's class information is available without locating the informative
region at all. If it does well, a method's success on that condition is not
evidence that the method found the region; if it is at chance, the condition
genuinely requires localisation.

This matters for the scale mechanism in particular. A burst of variance inside a
window also raises the variance of the whole series, so some of the signal is
visible globally, and the size of that leak has to be measured rather than
assumed.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["GlobalSummaryDiagnostic"]

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


class GlobalSummaryDiagnostic:
    """Extra trees on whole-series summary statistics.

    Nine quantiles, the mean and the standard deviation of each series, with no
    positional information of any kind.

    Parameters
    ----------
    n_estimators : int, default=200
        Size of the extra-trees ensemble.
    random_state : int or None, default=None
    """

    def __init__(self, n_estimators=200, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state

    @staticmethod
    def _features(X):
        # X arrives as (n_cases, 1, series_length) in the aeon convention.
        series = X[:, 0, :] if X.ndim == 3 else X
        quantiles = np.quantile(
            series, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], axis=1
        ).T
        return np.column_stack(
            [quantiles, series.mean(axis=1), series.std(axis=1)]
        )

    def fit(self, X, y):
        self._estimator = ExtraTreesClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self._estimator.fit(self._features(X), y)
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X):
        return self._estimator.predict(self._features(X))

    def predict_proba(self, X):
        return self._estimator.predict_proba(self._features(X))
