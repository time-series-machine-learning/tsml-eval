# Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb

# HYDRA: Competing convolutional kernels for fast and accurate time series classification
# https://arxiv.org/abs/2203.13652

# import copy
import time

import numpy as np
from aeon.regression.base import BaseRegressor
from aeon.utils.validation._dependencies import _check_soft_dependencies
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_check_soft_dependencies("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class Hydra(nn.Module):
    def __init__(self, ts_shape, k=8, g=64, max_num_channels=8):
        super().__init__()

        self.k = k  # num kernels per group
        self.g = g  # num groups

        _, self.n_dims_, self.series_length_ = ts_shape

        max_exponent = np.log2((self.series_length_ - 1) / (9 - 1))  # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div(
            (9 - 1) * self.dilations, 2, rounding_mode="floor"
        ).int()

        # if g > 1, assign: half the groups to X, half the groups to diff(X)
        divisor = 2 if self.g > 1 else 1
        _g = g // divisor
        self._g = _g

        self.W = [
            self.normalize(torch.randn(divisor, k * _g, 1, 9))
            for _ in range(self.num_dilations)
        ]

        # combine num_channels // 2 channels (2 < n < max_num_channels)
        if self.n_dims_ > 1:
            n_dims_per = np.clip(self.n_dims_ // 2, 2, max_num_channels)
            self.I = [
                torch.randint(0, self.n_dims_, (divisor, _g, n_dims_per))
                for _ in range(self.num_dilations)
            ]

    @staticmethod
    def normalize(W):
        W -= W.mean(-1, keepdims=True)
        W /= W.abs().sum(-1, keepdims=True)
        return W

    def forward(self, X):
        num_examples = X.shape[0]

        X = torch.from_numpy(X).float()

        if self.g > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):
            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            # diff_index == 0 -> X
            # diff_index == 1 -> diff(X)
            for diff_index in range(min(2, self.g)):
                if self.n_dims_ > 1:  # Multivariate
                    _Z = F.conv1d(
                        X[:, self.I[dilation_index][diff_index]].sum(2)
                        if diff_index == 0
                        else diff_X[:, self.I[dilation_index][diff_index]].sum(2),
                        self.W[dilation_index][diff_index],
                        dilation=d,
                        padding=p,
                        groups=self._g,
                    ).view(num_examples, self._g, self.k, -1)
                else:  # Univariate
                    _Z = F.conv1d(
                        X if diff_index == 0 else diff_X,
                        self.W[dilation_index][diff_index],
                        dilation=d,
                        padding=p,
                    ).view(num_examples, self._g, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self._g, self.k)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self._g, self.k)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z


class HydraPipeline:
    def __init__(self, ts_shape, k=8, g=64, max_num_channels=8):
        self.ts_shape = ts_shape
        self.k = k
        self.g = g
        self.max_num_channels = max_num_channels

    def fit(self, X, y=None):
        self.transformation = Hydra(
            self.ts_shape, self.k, self.g, self.max_num_channels
        )
        return self

    def transform(self, X):
        return self.transformation(X)


class HydraRegressor(BaseRegressor):
    """
    Hydra Regressor

    Examples
    --------
    >>> from tsml_eval.estimators.regression.convolution_based.hydra import HydraRegressor
    >>> from tsml.datasets import load_minimal_gas_prices
    >>> X, y = load_minimal_gas_prices()
    >>> regressor = HydraRegressor()
    >>> regressor = regressor.fit(X, y)
    >>> y_pred = regressor.predict(X)
    """
    _tags = {
        "capability:multivariate": True,
        "classifier_type": "kernel",
    }

    def __init__(self, k=8, g=64, max_num_channels=8, random_state=1, n_jobs=1):
        self.random_state = random_state
        if isinstance(random_state, int):
            torch.manual_seed(random_state)

        self.k = k
        self.g = g
        self.max_num_channels = max_num_channels

        self.n_jobs = n_jobs

        super().__init__()

    def _fit(self, X, y):
        self.regressor = make_pipeline(
            HydraPipeline(
                ts_shape=X.shape,
                k=self.k,
                g=self.g,
                max_num_channels=self.max_num_channels,
            ),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 10) * len(X)),
        )

        self.regressor.fit(X, y)

    def _predict(self, X):
        y_pred = self.regressor.predict(X)

        return y_pred
