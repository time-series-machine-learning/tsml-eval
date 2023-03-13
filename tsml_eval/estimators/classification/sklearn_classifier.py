# -*- coding: utf-8 -*-
"""A tsml wrapper for sklearn classifiers."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["SklearnToTsmlClassifier"]

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from tsml.base import BaseTimeSeriesEstimator, _clone_estimator


class SklearnToTsmlClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Wrapper for sklearn estimators."""

    def __init__(self, classifier, random_state=None):
        self.classifier = classifier
        self.random_state = random_state

        super(SklearnToTsmlClassifier, self).__init__()

    def fit(self, X, y):
        X, y = self._validate_data(X=X, y=y)

        check_classification_targets(y)

        if isinstance(X, np.ndarray):
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        else:
            raise ValueError("X must be a numpy array.")

        self._classifier = _clone_estimator(self.classifier, self.random_state)

        self._classifier.fit(X, y)

    def predict(self, X) -> np.ndarray:
        X = self._validate_data(X=X, reset=False)

        if isinstance(X, np.ndarray):
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        else:
            raise ValueError("X must be a numpy array.")

        return self._classifier.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        X = self._validate_data(X=X, reset=False)

        if isinstance(X, np.ndarray):
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        else:
            raise ValueError("X must be a numpy array.")

        return self._classifier.predict(X)
