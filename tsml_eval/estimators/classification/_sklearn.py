"""A wrapper for sklearn classifiers."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "SklearnToAeonClassifier",
]

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.transformations.collection.unequal_length import Padder
from aeon.utils.conversion import convert_collection
from aeon.utils.validation.collection import is_univariate
from sklearn.ensemble import RandomForestClassifier


class SklearnToAeonClassifier(BaseClassifier):
    """Wrapper for scikit-learn classifiers to use the aeon framework.

    Parameters
    ----------
    classifier : sklearn BaseEstimator
        A scikit-learn classifier object.
    pad_unequal : bool, default=False
        Whether to pad unequal length series to the same length before
        fitting/predicting.
        Cannot accept unequal length series if False.
    concatenate_channels : bool, default=False
        Whether to concatenate multivariate series into univariate series before
        fitting/predicting.
        Cannot accept multivariate series if False.
    random_state : int, RandomState instance or None, default=None
        Random state set when cloning the estimator. If None, no random
        state is set (but they may still be seeded prior to input).
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;

    Attributes
    ----------
    classifier_ : BaseEstimator
        The fitted scikit-learn classifier clone.
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(
        self,
        classifier,
        pad_unequal=False,
        concatenate_channels=False,
        random_state=None,
    ):
        self.classifier = classifier
        self.pad_unequal = pad_unequal
        self.concatenate_channels = concatenate_channels
        self.random_state = random_state

        super().__init__()
        self.set_tags(
            **{
                "capability:multivariate": concatenate_channels,
                "capability:unequal_length": pad_unequal,
            }
        )

    def _fit(self, X, y):
        X = self._check_and_convert(X)
        self.classifier_ = _clone_estimator(self.classifier, self.random_state)
        self.classifier_.fit(X, y)

    def _predict(self, X):
        X = self._check_and_convert(X)
        return self.classifier_.predict(X)

    def _predict_proba(self, X):
        X = self._check_and_convert(X)

        m = getattr(self.classifier_, "predict_proba", None)
        if callable(m):
            return m(X)

        return super()._predict_proba(X)

    def _check_and_convert(self, X):
        if self.pad_unequal:
            if not hasattr(self, "_padder"):
                self._padder = Padder()
                self._padder.fit(X)
            X = self._padder.transform(X)
        if self.concatenate_channels and not is_univariate(X):
            arr = np.empty((len(X), X[0].shape[0], X[0].shape[1]), dtype=X[0].dtype)
            for i, x in enumerate(X):
                arr[i, :, :] = x
            X = arr.reshape(arr.shape[0], -1)
        return convert_collection(X, "numpy2D")

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "classifier": RandomForestClassifier(n_estimators=5),
        }
