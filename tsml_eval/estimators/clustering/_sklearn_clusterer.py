"""A tsml wrapper for sklearn clusterers."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["SklearnToTsmlClusterer"]

import numpy as np
from aeon.base._base import _clone_estimator
from sklearn.base import ClusterMixin
from sklearn.utils.validation import check_is_fitted
from tsml.base import BaseTimeSeriesEstimator


class SklearnToTsmlClusterer(ClusterMixin, BaseTimeSeriesEstimator):
    """Wrapper for sklearn estimators to use the tsml base class."""

    def __init__(
        self,
        clusterer=None,
        pad_unequal=False,
        concatenate_channels=False,
        clone_estimator=True,
        random_state=None,
    ):
        self.clusterer = clusterer
        self.pad_unequal = pad_unequal
        self.concatenate_channels = concatenate_channels
        self.clone_estimator = clone_estimator
        self.random_state = random_state

        super().__init__()

    def fit(self, X, y=None):
        """Wrap fit."""
        if self.clusterer is None:
            raise ValueError("Clusterer not set")

        X = self._validate_data(
            X=X,
            ensure_univariate=not self.concatenate_channels,
            ensure_equal_length=not self.pad_unequal,
        )
        X = self._convert_X(
            X,
            pad_unequal=self.pad_unequal,
            concatenate_channels=self.concatenate_channels,
        )

        self._clusterer = (
            _clone_estimator(self.clusterer, self.random_state)
            if self.clone_estimator
            else self.clusterer
        )
        self._clusterer.fit(X, y)

        self.labels_ = self._clusterer.labels_

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

        return self._clusterer.predict(X)

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "equal_length_only": (False if self.pad_unequal else True),
            "univariate_only": False if self.concatenate_channels else True,
        }
