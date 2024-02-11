"""HYDRA classifier.

HYDRA: Competing convolutional kernels for fast and accurate time series classification
By Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb
https://arxiv.org/abs/2203.13652
"""

__author__ = ["patrickzib", "Arik Ermshaus"]
__all__ = ["HYDRA", "MultiRocketHydra"]

import numpy as np
from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.utils.validation import check_n_jobs
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class HYDRA(BaseClassifier):
    """
    Hydra Classifier.

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import HYDRA
    >>> from tsml.datasets import load_minimal_chinatown
    >>> X, y = load_minimal_chinatown()
    >>> classifier = HYDRA()
    >>> classifier = classifier.fit(X, y)
    >>> y_pred = classifier.predict(X)
    """

    _tags = {
        "capability:multithreading": False,
        "classifier_type": "convolution",
        "python_dependencies": "torch",
    }

    def __init__(self, k=8, g=64, n_jobs=1, random_state=None):
        self.k = k
        self.g = g
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        import torch

        from tsml_eval.estimators.classification.convolution_based._hydra_internal import (  # noqa
            HydraInternal,
        )

        if isinstance(self.random_state, int):
            torch.manual_seed(self.random_state)

        n_jobs = check_n_jobs(self.n_jobs)
        torch.set_num_threads(n_jobs)

        self.transform = HydraInternal(X.shape[-1])
        X_training_transform = self.transform(torch.tensor(X).float())

        self.clf = make_pipeline(
            SparseScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
        self.clf.fit(X_training_transform, y)

    def _predict(self, X) -> np.ndarray:
        import torch

        X_test_transform = self.transform(torch.tensor(X).float())
        return self.clf.predict(X_test_transform)


class MultiRocketHydra(BaseClassifier):
    """
    MultiRocket-Hydra Classifier.

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import (
    ...     MultiRocketHydra)
    >>> from tsml.datasets import load_minimal_chinatown
    >>> X, y = load_minimal_chinatown()
    >>> classifier = MultiRocketHydra()
    >>> classifier = classifier.fit(X, y)
    >>> y_pred = classifier.predict(X)
    """

    _tags = {
        "capability:multithreading": False,
        "classifier_type": "dictionary",
    }

    def __init__(self, k=8, g=64, n_jobs=1, random_state=None):
        self.k = k
        self.g = g
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        import torch

        from tsml_eval.estimators.classification.convolution_based._hydra_internal import (  # noqa
            HydraInternal,
        )

        if isinstance(self.random_state, int):
            torch.manual_seed(self.random_state)

        n_jobs = check_n_jobs(self.n_jobs)
        torch.set_num_threads(n_jobs)

        self.transform_hydra = HydraInternal(X.shape[-1])
        X_training_transform_hydra = self.transform_hydra(torch.tensor(X).float())

        self.transform_multirocket = MultiRocket()
        X_training_transform_multirocket = self.transform_multirocket.fit_transform(X)

        self.scale_hydra = SparseScaler()
        X_training_transform_hydra = self.scale_hydra.fit_transform(
            X_training_transform_hydra
        )

        self.scale_multirocket = StandardScaler()
        X_training_transform_multirocket = self.scale_multirocket.fit_transform(
            X_training_transform_multirocket
        )

        X_training_transform_concatenated = np.concatenate(
            (X_training_transform_hydra, X_training_transform_multirocket), axis=1
        )

        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.classifier.fit(X_training_transform_concatenated, y)

    def _predict(self, X) -> np.ndarray:
        import torch

        X_test_transform_hydra = self.transform_hydra(torch.tensor(X).float())
        X_test_transform_multirocket = self.transform_multirocket.transform(X)

        X_test_transform_hydra = self.scale_hydra.transform(X_test_transform_hydra)
        X_test_transform_multirocket = self.scale_multirocket.transform(
            X_test_transform_multirocket
        )

        X_test_transform_concatenated = np.concatenate(
            (X_test_transform_hydra, X_test_transform_multirocket), axis=1
        )

        return self.classifier.predict(X_test_transform_concatenated)


class SparseScaler:
    def __init__(self, mask=True, exponent=4):
        self.mask = mask
        self.exponent = exponent

        self.fitted = False

    def fit(self, X, y=None):
        assert not self.fitted, "Already fitted."

        X = X.clamp(0).sqrt()

        self.epsilon = (X == 0).float().mean(0) ** self.exponent + 1e-8

        self.mu = X.mean(0)
        self.sigma = X.std(0) + self.epsilon

        self.fitted = True

    def transform(self, X):
        assert self.fitted, "Not fitted."

        X = X.clamp(0).sqrt()

        if self.mask:
            return ((X - self.mu) * (X != 0)) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X, y=None):
        self.fit(X)

        return self.transform(X)
