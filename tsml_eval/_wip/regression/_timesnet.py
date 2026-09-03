"""Work-in-progress sequence-to-scalar regression adaptation of TimesNet."""

from __future__ import annotations

__maintainer__ = ["TonyBagnall"]
__all__ = ["TimesNetRegressor"]

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from aeon.regression.base import BaseRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from tsml_eval._wip.classification._timesnet import (
    _StandardisePerChannel,
    _TimesNetClassificationModel,
    _set_torch_seed,
)


class TimesNetRegressor(BaseRegressor):
    """TimesNet regressor for one continuous target per time series.

    This retains the vendored TimesNet classification backbone and replaces
    its class projection with a single linear output trained by mean squared
    error. Input uses aeon's ``(n_cases, n_channels, n_timepoints)`` layout.

    Input-channel and target standardisation are learned only from the internal
    training subset. The validation subset is used solely for checkpoint
    selection and early stopping.

    Parameters
    ----------
    e_layers : int, default=2
        Number of TimesBlocks.
    d_model : int, default=32
        Embedding dimension.
    d_ff : int, default=64
        Hidden dimension in the inception-style convolution blocks.
    top_k : int, default=3
        Number of dominant FFT periods selected in each TimesBlock.
    num_kernels : int, default=6
        Number of inception kernels per block.
    dropout : float, default=0.1
        Dropout rate.
    batch_size : int, default=16
        Training and prediction batch size.
    n_epochs : int, default=30
        Maximum number of training epochs.
    learning_rate : float, default=0.001
        Learning rate for RAdam.
    lr_adjust : {"type1", "cosine", None}, default=None
        Optional learning-rate schedule from the original implementation.
    patience : int, default=10
        Early-stopping patience based on validation MSE.
    validation_size : float, default=0.2
        Fraction of training cases held out for validation.
    gradient_clip : float or None, default=4.0
        Maximum gradient norm. No clipping is applied when ``None``.
    standardise : bool, default=True
        Whether to standardise each input channel.
    standardise_y : bool, default=True
        Whether to standardise the target using the internal training subset.
    device : str or None, default=None
        Torch device. ``None`` selects CUDA when available, otherwise CPU.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Whether to print training progress.

    References
    ----------
    .. [1] Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long, M.
       "TimesNet: Temporal 2D-Variation Modeling for General Time Series
       Analysis." International Conference on Learning Representations, 2023.
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
        e_layers: int = 2,
        d_model: int = 32,
        d_ff: int = 64,
        top_k: int = 3,
        num_kernels: int = 6,
        dropout: float = 0.1,
        batch_size: int = 16,
        n_epochs: int = 30,
        learning_rate: float = 1e-3,
        lr_adjust: str | None = None,
        patience: int = 10,
        validation_size: float = 0.2,
        gradient_clip: float | None = 4.0,
        standardise: bool = True,
        standardise_y: bool = True,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.e_layers = e_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.lr_adjust = lr_adjust
        self.patience = patience
        self.validation_size = validation_size
        self.gradient_clip = gradient_clip
        self.standardise = standardise
        self.standardise_y = standardise_y
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        super().__init__()

    def _validate_parameters(self) -> None:
        for name in [
            "e_layers",
            "d_model",
            "d_ff",
            "top_k",
            "num_kernels",
            "batch_size",
            "n_epochs",
            "patience",
        ]:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not 0 <= self.validation_size < 1:
            raise ValueError("validation_size must be in [0, 1)")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive or None")
        if not isinstance(self.standardise, (bool, np.bool_)):
            raise ValueError("standardise must be a boolean")
        if not isinstance(self.standardise_y, (bool, np.bool_)):
            raise ValueError("standardise_y must be a boolean")
        if self.lr_adjust not in (None, "none", "type1", "cosine"):
            raise ValueError(
                'lr_adjust must be one of "type1", "cosine", or None, got '
                f"{self.lr_adjust!r}"
            )

    def _resolve_device(self) -> torch.device:
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(self.device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved

    @staticmethod
    def _preprocess_X(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_t = np.transpose(X, (0, 2, 1)).astype(np.float32, copy=False)
        mask = np.ones((X.shape[0], X.shape[2]), dtype=np.float32)
        return x_t, mask

    def _make_loader(self, x, mask, y=None, shuffle=False):
        x_tensor = torch.from_numpy(x)
        mask_tensor = torch.from_numpy(mask)
        if y is None:
            dataset = torch.utils.data.TensorDataset(x_tensor, mask_tensor)
        else:
            y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))
            dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, mask_tensor)
        generator = torch.Generator()
        generator.manual_seed(self.random_state_)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def _adjust_learning_rate(self, optimiser, epoch):
        if self.lr_adjust in (None, "none") or epoch % 5 != 0:
            return None
        if self.lr_adjust == "type1":
            lr = self.learning_rate * (0.5 ** (epoch - 1))
        else:
            lr = (
                self.learning_rate
                / 2
                * (1 + math.cos(epoch / self.n_epochs * math.pi))
            )
        for param_group in optimiser.param_groups:
            param_group["lr"] = lr
        return lr

    def _loader_mse(self, loader):
        self.network_.eval()
        squared_error = 0.0
        n_cases = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_mask in loader:
                predictions = self.network_(
                    batch_x.to(self.device_), batch_mask.to(self.device_)
                ).squeeze(-1)
                errors = nn.functional.mse_loss(
                    predictions,
                    batch_y.to(self.device_),
                    reduction="none",
                )
                squared_error += errors.sum().item()
                n_cases += len(batch_y)
        return squared_error / n_cases

    def _fit(self, X: np.ndarray, y):
        self._validate_parameters()
        rng = check_random_state(self.random_state)
        self.random_state_ = int(rng.randint(np.iinfo(np.int32).max))
        _set_torch_seed(self.random_state_)
        self.device_ = self._resolve_device()
        self.n_channels_, self.seq_len_ = X.shape[1], X.shape[2]

        x_t, mask = self._preprocess_X(X)
        y = np.asarray(y, dtype=np.float32)
        if self.validation_size > 0 and len(X) >= 2:
            train_idx, val_idx = train_test_split(
                np.arange(len(X)),
                test_size=self.validation_size,
                random_state=self.random_state_,
            )
        else:
            train_idx = np.arange(len(X))
            val_idx = np.array([], dtype=int)

        if self.standardise:
            self.scaler_ = _StandardisePerChannel().fit(x_t[train_idx])
            x_t = self.scaler_.transform(x_t)
        else:
            self.scaler_ = None

        if self.standardise_y:
            self.y_mean_ = float(np.mean(y[train_idx], dtype=np.float64))
            y_scale = float(np.std(y[train_idx], dtype=np.float64))
            self.y_scale_ = y_scale if y_scale > 0 else 1.0
        else:
            self.y_mean_, self.y_scale_ = 0.0, 1.0
        y = (y - self.y_mean_) / self.y_scale_

        self.network_ = _TimesNetClassificationModel(
            seq_len=self.seq_len_,
            enc_in=self.n_channels_,
            num_class=1,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            top_k=self.top_k,
            num_kernels=self.num_kernels,
            dropout=self.dropout,
        ).to(self.device_)
        self.optimizer_ = torch.optim.RAdam(
            self.network_.parameters(), lr=self.learning_rate
        )
        train_loader = self._make_loader(
            x_t[train_idx], mask[train_idx], y[train_idx], shuffle=True
        )
        val_loader = (
            None
            if len(val_idx) == 0
            else self._make_loader(
                x_t[val_idx], mask[val_idx], y[val_idx], shuffle=False
            )
        )

        best_state = deepcopy(self.network_.state_dict())
        best_loss = math.inf
        epochs_without_improvement = 0
        self.history_ = []
        for epoch in range(self.n_epochs):
            self.network_.train()
            total_loss = 0.0
            n_train = 0
            for batch_x, batch_y, batch_mask in train_loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                batch_mask = batch_mask.to(self.device_)
                self.optimizer_.zero_grad()
                predictions = self.network_(batch_x, batch_mask).squeeze(-1)
                loss = nn.functional.mse_loss(predictions, batch_y)
                loss.backward()
                if self.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.network_.parameters(), max_norm=self.gradient_clip
                    )
                self.optimizer_.step()
                total_loss += loss.item() * len(batch_y)
                n_train += len(batch_y)
            train_loss = total_loss / n_train
            validation_loss = (
                train_loss if val_loader is None else self._loader_mse(val_loader)
            )
            self.history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = deepcopy(self.network_.state_dict())
                self.best_epoch_ = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    val_loader is not None
                    and epochs_without_improvement >= self.patience
                ):
                    break
            new_lr = self._adjust_learning_rate(self.optimizer_, epoch + 1)
            if self.verbose:
                print(
                    f"epoch={epoch + 1} train_loss={train_loss:.6f} "
                    f"val_loss={validation_loss:.6f}"
                )
                if new_lr is not None:
                    print(f"updating learning rate to {new_lr}")

        self.network_.load_state_dict(best_state)
        self.network_.eval()
        self.best_validation_loss_ = best_loss
        return self

    def _check_shape(self, X):
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"X has {X.shape[1]} channels, but the regressor was fitted "
                f"with {self.n_channels_}."
            )
        if X.shape[2] != self.seq_len_:
            raise ValueError(
                f"X has length {X.shape[2]}, but the regressor was fitted with "
                f"length {self.seq_len_}."
            )

    def _predict(self, X: np.ndarray):
        self._check_shape(X)
        x_t, mask = self._preprocess_X(X)
        if self.scaler_ is not None:
            x_t = self.scaler_.transform(x_t)
        loader = self._make_loader(x_t, mask, shuffle=False)
        predictions = []
        self.network_.eval()
        with torch.no_grad():
            for batch_x, batch_mask in loader:
                output = self.network_(
                    batch_x.to(self.device_), batch_mask.to(self.device_)
                ).squeeze(-1)
                predictions.append(output.cpu().numpy())
        scaled = np.concatenate(predictions)
        return scaled * self.y_scale_ + self.y_mean_

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict:
        """Return a small parameter set for aeon estimator checks."""
        return {
            "e_layers": 1,
            "d_model": 8,
            "d_ff": 8,
            "top_k": 2,
            "num_kernels": 2,
            "n_epochs": 2,
            "batch_size": 4,
            "patience": 2,
            "validation_size": 0.2,
            "device": "cpu",
            "random_state": 0,
        }
