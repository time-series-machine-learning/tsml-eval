"""Work-in-progress regression adapter for the vendored TS2Vec encoder."""

from __future__ import annotations

__maintainer__ = ["TonyBagnall"]
__all__ = ["TS2VecRegressor"]

import numpy as np
from aeon.regression.base import BaseRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


class TS2VecRegressor(BaseRegressor):
    """TS2Vec self-supervised encoder followed by a ridge regression probe.

    The vendored TS2Vec encoder is trained exclusively on the supplied training
    collection using the authors' hierarchical contrastive objective. A
    standardized ``RidgeCV`` probe is then fitted to the training embeddings.
    Test cases participate only in encoding after both components are fitted.

    Parameters
    ----------
    output_dims : int, default=320
        Width of the learned representation.
    hidden_dims : int, default=64
        Width of the encoder hidden layers.
    depth : int, default=10
        Number of dilated convolution blocks.
    n_iters : int or None, default=None
        Number of self-supervised training iterations.
    n_epochs : int or None, default=None
        Number of self-supervised epochs, as an alternative to ``n_iters``.
    batch_size : int, default=16
        Encoder training batch size.
    learning_rate : float, default=0.001
        Encoder learning rate.
    max_train_length : int, default=3000
        Maximum section length used during encoder training.
    temporal_unit : int, default=0
        Minimum unit for the temporal contrastive objective.
    alphas : tuple of float, default=(0.01, 0.1, 1.0, 10.0, 100.0)
        Regularization strengths considered by ``RidgeCV``.
    probe_max_samples : int or None, default=None
        Optional cap on cases used to fit the regression probe. Subsampling is
        random and uses training cases only. ``None`` uses every case.
    device : {"auto", "cpu", "cuda"} or torch device string, default="auto"
        Device used for encoder training and encoding.
    verbose : bool, default=False
        Whether to print encoder training progress.
    random_state : int, RandomState instance or None, default=1234
        Seed controlling encoder training and optional probe subsampling.

    References
    ----------
    .. [1] Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y. and Xu, B.
       "TS2Vec: Towards Universal Representation of Time Series." AAAI, 2022.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
        n_iters: int | None = None,
        n_epochs: int | None = None,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        max_train_length: int = 3000,
        temporal_unit: int = 0,
        alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
        probe_max_samples: int | None = None,
        device: str = "auto",
        verbose: bool = False,
        random_state=1234,
    ):
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.n_iters = n_iters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.alphas = alphas
        self.probe_max_samples = probe_max_samples
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _validate_parameters(self):
        for name in [
            "output_dims",
            "hidden_dims",
            "depth",
            "batch_size",
            "max_train_length",
        ]:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ["n_iters", "n_epochs"]:
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not isinstance(self.temporal_unit, int) or self.temporal_unit < 0:
            raise ValueError("temporal_unit must be a non-negative integer")
        if self.probe_max_samples is not None and (
            not isinstance(self.probe_max_samples, int)
            or self.probe_max_samples <= 0
        ):
            raise ValueError("probe_max_samples must be a positive integer or None")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must contain positive values")

    def _resolve_device(self):
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return self.device

    @staticmethod
    def _to_original_layout(X):
        return np.transpose(np.asarray(X, dtype=np.float32), (0, 2, 1))

    def _encode(self, X):
        return self.encoder_.encode(
            self._to_original_layout(X), encoding_window="full_series"
        )

    def _subsample(self, features, y, rng):
        if self.probe_max_samples is None or len(features) <= self.probe_max_samples:
            return features, y
        indices = rng.choice(
            len(features), size=self.probe_max_samples, replace=False
        )
        return features[indices], y[indices]

    def _fit(self, X, y):
        self._validate_parameters()

        import torch

        from tsml_eval._wip.classification._ts2vec_original.ts2vec import TS2Vec

        rng = check_random_state(self.random_state)
        self.random_state_ = int(rng.randint(np.iinfo(np.int32).max))
        np.random.seed(self.random_state_)
        torch.manual_seed(self.random_state_)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state_)

        self.device_ = self._resolve_device()
        self.n_channels_, self.n_timepoints_ = X.shape[1], X.shape[2]
        self.encoder_ = TS2Vec(
            input_dims=self.n_channels_,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            device=self.device_,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            max_train_length=self.max_train_length,
            temporal_unit=self.temporal_unit,
        )
        n_iters = self.n_iters
        if n_iters is None and self.n_epochs is None:
            n_iters = 200 if X.shape[0] <= 100 else 600
        self.loss_log_ = self.encoder_.fit(
            self._to_original_layout(X),
            n_epochs=self.n_epochs,
            n_iters=n_iters,
            verbose=self.verbose,
        )

        features, probe_y = self._subsample(
            self._encode(X), np.asarray(y, dtype=np.float64), rng
        )
        self.probe_cases_ = len(features)
        self.probe_ = make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.asarray(self.alphas, dtype=float))
        ).fit(features, probe_y)
        return self

    def _check_shape(self, X):
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"X has {X.shape[1]} channels, but the regressor was fitted "
                f"with {self.n_channels_}."
            )
        if X.shape[2] != self.n_timepoints_:
            raise ValueError(
                f"X has length {X.shape[2]}, but the regressor was fitted with "
                f"length {self.n_timepoints_}."
            )

    def _predict(self, X):
        self._check_shape(X)
        return self.probe_.predict(self._encode(X))

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict:
        """Return a small parameter set for aeon estimator checks."""
        return {
            "output_dims": 8,
            "hidden_dims": 8,
            "depth": 2,
            "n_iters": 2,
            "batch_size": 4,
            "device": "cpu",
            "random_state": 0,
        }
