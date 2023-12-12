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
from aeon.utils.validation._dependencies import _check_soft_dependencies
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_check_soft_dependencies("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class HYDRA(BaseClassifier):
    """
    Hydra Classifier

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
        "classifier_type": "dictionary",
    }

    def __init__(self, k=8, g=64, n_jobs=1, random_state=None):
        self.k = k
        self.g = g
        self.n_jobs = n_jobs
        self.random_state = random_state

        if isinstance(random_state, int):
            torch.manual_seed(random_state)

        n_jobs = check_n_jobs(n_jobs)
        torch.set_num_threads(n_jobs)

        super(HYDRA, self).__init__()

    def _fit(self, X, y):
        self.transform = HydraInternal(X.shape[-1])
        X_training_transform = self.transform(torch.tensor(X).float())

        self.clf = make_pipeline(
            SparseScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
        self.clf.fit(X_training_transform, y)

    def _predict(self, X) -> np.ndarray:
        X_test_transform = self.transform(torch.tensor(X).float())
        return self.clf.predict(X_test_transform)


class MultiRocketHydra(BaseClassifier):
    """
    MultiRocket-Hydra Classifier

    Examples
    --------
    >>> from tsml_eval.estimators.classification.convolution_based import MultiRocketHydra
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

        if isinstance(random_state, int):
            torch.manual_seed(random_state)

        n_jobs = check_n_jobs(n_jobs)
        torch.set_num_threads(n_jobs)

        super(MultiRocketHydra, self).__init__()

    def _fit(self, X, y):
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


class HydraInternal(nn.Module):
    """HYDRA classifier."""

    def __init__(self, input_length, k=8, g=64):
        super().__init__()

        self.k = k  # num kernels per group
        self.g = g  # num groups

        max_exponent = np.log2((input_length - 1) / (9 - 1))  # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div(
            (9 - 1) * self.dilations, 2, rounding_mode="floor"
        ).int()

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        self.W = torch.randn(self.num_dilations, self.divisor, self.k * self.h, 1, 9)
        self.W = self.W - self.W.mean(-1, keepdims=True)
        self.W = self.W / self.W.abs().sum(-1, keepdims=True)

    # transform in batches of *batch_size*
    def batch(self, X, batch_size=256):
        """Batches."""
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for batch in batches:
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X):
        """Forward."""
        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):
            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):
                _Z = F.conv1d(
                    X if diff_index == 0 else diff_X,
                    self.W[dilation_index, diff_index],
                    dilation=d,
                    padding=p,
                ).view(num_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.k)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.k)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z


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
