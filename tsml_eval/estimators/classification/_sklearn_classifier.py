"""A tsml wrapper for sklearn classifiers."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["SklearnToTsmlClassifier"]

import numpy as np
from aeon.base._base import _clone_estimator
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from tsml.base import BaseTimeSeriesEstimator


class SklearnToTsmlClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Wrapper for sklearn estimators to use the tsml base class."""

    def __init__(
        self,
        classifier=None,
        pad_unequal=False,
        concatenate_channels=False,
        clone_estimator=True,
        random_state=None,
    ):
        self.classifier = classifier
        self.pad_unequal = pad_unequal
        self.concatenate_channels = concatenate_channels
        self.clone_estimator = clone_estimator
        self.random_state = random_state

        super().__init__()

    def fit(self, X, y):
        """Wrap fit."""
        if self.classifier is None:
            raise ValueError("Classifier not set")

        X, y = self._validate_data(
            X=X,
            y=y,
            ensure_univariate=not self.concatenate_channels,
            ensure_equal_length=not self.pad_unequal,
        )
        X = self._convert_X(
            X,
            pad_unequal=self.pad_unequal,
            concatenate_channels=self.concatenate_channels,
        )

        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self._classifier = (
            _clone_estimator(self.classifier, self.random_state)
            if self.clone_estimator
            else self.classifier
        )
        self._classifier.fit(X, y)

        return self

    def predict(self, X) -> np.ndarray:
        """Wrap predict."""
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(
            X,
            pad_unequal=self.pad_unequal,
            concatenate_channels=self.concatenate_channels,
        )

        return self._classifier.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Wrap predict_proba."""
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(
            X,
            pad_unequal=self.pad_unequal,
            concatenate_channels=self.concatenate_channels,
        )

        return self._classifier.predict_proba(X)

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "equal_length_only": (False if self.pad_unequal else True),
            "univariate_only": False if self.concatenate_channels else True,
        }
