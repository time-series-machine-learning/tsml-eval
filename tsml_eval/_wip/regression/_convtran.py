"""Work-in-progress regression adaptation of ConvTran.

The feature extractor is the direct PyTorch port of the original ConvTran
classification network in :mod:`tsml_eval._wip.classification._convtran`.
This estimator replaces its classification output with a scalar linear head
and trains it using mean squared error.
"""

__maintainer__ = []
__all__ = ["ConvTranRegressor"]

import math
from copy import deepcopy

import numpy as np
import torch
from aeon.regression.base import BaseRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import check_random_state
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tsml_eval._wip.classification._convtran import _ConvTranNetwork, _RAdam


class ConvTranRegressor(BaseRegressor):
    """Convolutional Transformer (ConvTran) regressor.

    This adapts the original ConvTran classification architecture to
    sequence-to-scalar regression. The ConvTran backbone is unchanged, while
    its final linear layer has one output and is trained with mean squared
    error. Input follows aeon's ``(n_cases, n_channels, n_timepoints)`` layout.

    Parameters
    ----------
    emb_size : int, default=16
        Transformer embedding dimension. Must be even and divisible by
        ``num_heads``.
    dim_ff : int, default=256
        Hidden dimension of the feed-forward block.
    num_heads : int, default=8
        Number of attention heads.
    dropout : float, default=0.01
        Dropout used by tAPE and the feed-forward block.
    n_epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=16
        Training and prediction batch size.
    learning_rate : float, default=0.001
        Learning rate for the authors' RAdam optimizer.
    validation_size : float, default=0.2
        Fraction of the supplied training cases held out for validation and
        best-loss checkpoint selection. Set to 0 to train on all cases and
        retain the epoch with the lowest training loss.
    standardize_y : bool, default=True
        Whether to standardize the regression target. The mean and standard
        deviation are learned from the internal training subset only, then
        predictions are transformed back to the original target scale.
    gradient_clip_norm : float or None, default=4.0
        Maximum gradient norm, matching the original ConvTran training loop.
        No clipping is performed when ``None``.
    device : {"auto", "cpu", "cuda"} or torch device string, default="auto"
        Device used for training and prediction. ``"auto"`` selects CUDA when
        available and otherwise CPU.
    num_workers : int, default=0
        Number of data-loader worker processes.
    verbose : bool, default=False
        Whether to print epoch losses.
    random_state : int, RandomState instance or None, default=1234
        Seed controlling the validation split, model initialization, and batch
        shuffling. The default matches the original ConvTran implementation.

    Attributes
    ----------
    model_ : torch.nn.Module
        Fitted ConvTran network restored to the epoch with the best validation
        loss.
    history_ : list of dict
        Training and validation mean squared error for each epoch, measured on
        the target scale used for optimization.
    device_ : torch.device
        Resolved training device.
    best_epoch_ : int
        One-based index of the retained epoch.
    best_validation_loss_ : float
        Loss used to select the retained epoch.
    y_mean_ : float
        Training-subset target mean used for standardization.
    y_scale_ : float
        Training-subset target standard deviation used for standardization.

    References
    ----------
    .. [1] Foumani, N. M., Tan, C. W., Webb, G. I., and Salehi, M.
       "Improving position encoding of transformers for multivariate time
       series classification." Data Mining and Knowledge Discovery 38,
       22-48, 2024.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        emb_size=16,
        dim_ff=256,
        num_heads=8,
        dropout=0.01,
        n_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
        validation_size=0.2,
        standardize_y=True,
        gradient_clip_norm=4.0,
        device="auto",
        num_workers=0,
        verbose=False,
        random_state=1234,
    ):
        self.emb_size = emb_size
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_size = validation_size
        self.standardize_y = standardize_y
        self.gradient_clip_norm = gradient_clip_norm
        self.device = device
        self.num_workers = num_workers
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _validate_parameters(self):
        if not isinstance(self.emb_size, int) or self.emb_size <= 0:
            raise ValueError("emb_size must be a positive integer")
        if self.emb_size % 2 != 0:
            raise ValueError("emb_size must be even for tAPE")
        if not isinstance(self.num_heads, int) or self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if self.emb_size % self.num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads")
        if not isinstance(self.dim_ff, int) or self.dim_ff <= 0:
            raise ValueError("dim_ff must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.n_epochs, int) or self.n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not 0 <= self.validation_size < 1:
            raise ValueError("validation_size must be in [0, 1)")
        if not isinstance(self.standardize_y, (bool, np.bool_)):
            raise ValueError("standardize_y must be a boolean")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")

    def _resolve_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(self.device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved

    def _seed_torch(self):
        rng = check_random_state(self.random_state)
        self.random_state_ = int(rng.randint(np.iinfo(np.int32).max))
        torch.manual_seed(self.random_state_)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state_)

    def _make_loader(self, X, y=None, shuffle=False):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        if y is None:
            dataset = TensorDataset(X_tensor)
        else:
            y_tensor = torch.as_tensor(y, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)
        generator = torch.Generator()
        generator.manual_seed(self.random_state_)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=self.device_.type == "cuda",
            num_workers=self.num_workers,
            generator=generator,
        )

    def _loss(self, loader, train):
        self.model_.train(mode=train)
        total_loss = 0.0
        total_cases = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                y_batch = y_batch.to(self.device_, non_blocking=True)
                predictions = self.model_(X_batch).squeeze(-1)
                losses = nn.functional.mse_loss(
                    predictions, y_batch, reduction="none"
                )
                loss = losses.mean()
                if train:
                    self.optimizer_.zero_grad()
                    loss.backward()
                    if self.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.model_.parameters(), self.gradient_clip_norm
                        )
                    self.optimizer_.step()
                total_loss += losses.detach().sum().item()
                total_cases += len(y_batch)
        return total_loss / total_cases

    def _fit(self, X, y):
        self._validate_parameters()
        self._seed_torch()
        self.device_ = self._resolve_device()
        self.n_channels_ = X.shape[1]
        self.n_timepoints_ = X.shape[2]
        y = np.asarray(y, dtype=np.float32)

        if self.validation_size > 0:
            splitter = ShuffleSplit(
                n_splits=1,
                test_size=self.validation_size,
                random_state=self.random_state_,
            )
            train_indices, validation_indices = next(splitter.split(X))
            X_train, y_train = X[train_indices], y[train_indices]
            X_validation, y_validation = X[validation_indices], y[validation_indices]
        else:
            X_train, y_train = X, y
            X_validation = y_validation = None

        if self.standardize_y:
            self.y_mean_ = float(np.mean(y_train, dtype=np.float64))
            y_scale = float(np.std(y_train, dtype=np.float64))
            self.y_scale_ = y_scale if y_scale > 0 else 1.0
        else:
            self.y_mean_ = 0.0
            self.y_scale_ = 1.0

        y_train = (y_train - self.y_mean_) / self.y_scale_
        if y_validation is not None:
            y_validation = (y_validation - self.y_mean_) / self.y_scale_

        self.model_ = _ConvTranNetwork(
            n_channels=self.n_channels_,
            n_timepoints=self.n_timepoints_,
            n_classes=1,
            emb_size=self.emb_size,
            num_heads=self.num_heads,
            dim_ff=self.dim_ff,
            dropout=self.dropout,
        ).to(self.device_)
        self.optimizer_ = _RAdam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=0
        )
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        validation_loader = (
            None
            if X_validation is None
            else self._make_loader(X_validation, y_validation, shuffle=False)
        )

        best_state = None
        best_loss = math.inf
        self.history_ = []
        for epoch in range(self.n_epochs):
            train_loss = self._loss(train_loader, train=True)
            validation_loss = (
                train_loss
                if validation_loader is None
                else self._loss(validation_loader, train=False)
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
                self.best_epoch_ = epoch + 1
                best_state = deepcopy(self.model_.state_dict())
            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs}: "
                    f"loss={train_loss:.6f}, val_loss={validation_loss:.6f}"
                )

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_validation_loss_ = best_loss
        return self

    def _predict(self, X):
        loader = self._make_loader(X, shuffle=False)
        self.model_.eval()
        predictions = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                predictions.append(self.model_(X_batch).squeeze(-1).cpu())
        scaled = torch.cat(predictions, dim=0).numpy()
        return scaled * self.y_scale_ + self.y_mean_

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return a small parameter set for aeon estimator checks."""
        return {
            "emb_size": 8,
            "dim_ff": 16,
            "num_heads": 2,
            "n_epochs": 2,
            "batch_size": 4,
            "validation_size": 0.25,
            "device": "cpu",
            "random_state": 0,
        }
