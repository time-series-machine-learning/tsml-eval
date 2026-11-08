"""Level-1 (stacking) models for HC2 component probabilities.

Input is a flat stacking problem from extract_stacking_data.py, (n_cases,
n_components * n_classes), one component's class probabilities after another.
Column j of a component is class index j; the labels arrive as strings of those
indices.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


class CAWPE(ClassifierMixin, BaseEstimator):
    """The HC2 combination: weights train accuracy**alpha, optional prior correction.

    alpha=0 is the unweighted mean of the component probabilities.
    """

    def __init__(self, alpha=4, prior_correction=0.0):
        self.alpha = alpha
        self.prior_correction = prior_correction

    def _components(self, X):
        k = len(self.classes_)
        if X.shape[1] % k:
            raise ValueError(f"{X.shape[1]} features is not a multiple of {k} classes")
        # reorder each component's columns from class index to classes_ order
        return np.asarray(X).reshape(len(X), -1, k)[:, :, self._columns]

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._columns = [int(c) for c in self.classes_]
        P = self._components(X)
        yi = np.searchsorted(self.classes_, y)
        acc = (P.argmax(axis=2) == yi[:, None]).mean(axis=0)
        self.weights_ = acc**self.alpha
        self.class_prior_ = np.bincount(yi, minlength=len(self.classes_)) / len(yi)
        return self

    def predict_proba(self, X):
        d = np.einsum("nmk,m->nk", self._components(X), self.weights_)
        if self.prior_correction:
            d = d / self.class_prior_**self.prior_correction
        s = d.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        return d / s

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _log(X):
    return np.log(np.clip(X, 1e-4, 1))


def stackers(random_state=0, foundation=False, device=None):
    """{name: (category, estimator)} for the classical or the foundation models."""
    if foundation:
        from foundation_models import foundation_stackers

        return foundation_stackers(random_state, device=device)
    rf = dict(n_estimators=500, random_state=random_state, n_jobs=1)
    return {
        "Mean": ("Reference", CAWPE(alpha=0)),
        "CAWPE": ("Reference", CAWPE(alpha=4)),
        "CAWPE-p0.5": ("Reference", CAWPE(alpha=4, prior_correction=0.5)),
        "LR": ("Linear", LogisticRegression(C=1.0, max_iter=2000)),
        "LR-log": (
            "Linear",
            make_pipeline(
                FunctionTransformer(_log), LogisticRegression(C=1.0, max_iter=2000)
            ),
        ),
        "LDA": (
            "Linear",
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ),
        "NB": ("Other", GaussianNB()),
        "RF": ("Trees", RandomForestClassifier(**rf)),
        "ET": ("Trees", ExtraTreesClassifier(**rf)),
        "HGB": (
            "Trees",
            HistGradientBoostingClassifier(random_state=random_state),
        ),
    }
