"""Adapters for classifiers implemented in sktime."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["SktimeToAeonClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from aeon.utils.conversion import convert_collection
from sklearn.base import clone


class SktimeToAeonClassifier(BaseClassifier):
    """Run an sktime classifier through the aeon classifier interface.

    The adapter deliberately keeps the input multivariate. The generic sklearn
    adapter is unsuitable here because it flattens multivariate collections.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
    }

    def __init__(self, classifier, capability_multivariate=True):
        self.classifier = classifier
        self.capability_multivariate = capability_multivariate
        super().__init__()
        self.set_tags(**{"capability:multivariate": capability_multivariate})

    def _fit(self, X, y):
        X = convert_collection(X, "numpy3D")
        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X, y)
        return self

    def _predict(self, X):
        X = convert_collection(X, "numpy3D")
        return self.classifier_.predict(X)

    def _predict_proba(self, X):
        X = convert_collection(X, "numpy3D")
        probabilities = self.classifier_.predict_proba(X)

        # Align sktime's probability columns with aeon's classes_ if needed.
        sktime_classes = getattr(self.classifier_, "classes_", self.classes_)
        if not np.array_equal(sktime_classes, self.classes_):
            aligned = np.zeros((len(X), len(self.classes_)), dtype=probabilities.dtype)
            for source, target in enumerate(self.classes_):
                matches = np.flatnonzero(np.asarray(sktime_classes) == target)
                if len(matches) == 1:
                    aligned[:, source] = probabilities[:, matches[0]]
            probabilities = aligned
        return probabilities

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return test parameters when sktime is installed."""
        from sktime.classification.deep_learning import CNNClassifier

        return {"classifier": CNNClassifier(n_epochs=1, batch_size=2)}
