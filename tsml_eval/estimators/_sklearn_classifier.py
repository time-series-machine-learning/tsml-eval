# -*- coding: utf-8 -*-
"""A tsml wrapper for sklearn classifiers."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["SklearnToTsmlClassifier"]

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from tsml.base import BaseTimeSeriesEstimator, _clone_estimator


class SklearnToTsmlClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Wrapper for sklearn estimators."""

    def __init__(self, classifier=None, concatenate_channels=False, random_state=None):
        self.classifier = classifier
        self.concatenate_channels = concatenate_channels

        self.random_state = random_state

        super(SklearnToTsmlClassifier, self).__init__()

    def fit(self, X, y):
        if self.classifier is None:
            raise ValueError("Classifier not set")

        X, y = self._validate_data(X=X, y=y)
        X = self._convert_X(X, concatenate_channels=self.concatenate_channels)

        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self._classifier = _clone_estimator(self.classifier, self.random_state)
        self._classifier.fit(X, y)

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X, concatenate_channels=self.concatenate_channels)

        return self._classifier.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X, concatenate_channels=self.concatenate_channels)

        return self._classifier.predict_proba(X)

    def _more_tags(self):
        return {"X_types": ["2darray"]}
