"""Work-in-progress aeon classifier port of the original ConvTran code.

This module is adapted directly from commit 148afb6 of
https://github.com/Navidfoumani/ConvTran.

Copyright (c) 2022 Department of Data Science and Artificial Intelligence
@Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__maintainer__ = []
__all__ = ["ConvTranClassifier"]

import math
from copy import deepcopy

import numpy as np
import torch
from aeon.classification import BaseClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_random_state
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset


class _TimeAbsolutePositionalEncoding(nn.Module):
    """Original time absolute positional encoding (tAPE)."""

    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        scale = d_model / max_len
        positional_encoding[:, 0::2] = torch.sin(position * divisor * scale)
        positional_encoding[:, 1::2] = torch.cos(position * divisor * scale)
        self.register_buffer("pe", positional_encoding.unsqueeze(0))

    def forward(self, x):
        """Add tAPE to input shaped (batch, timepoints, embedding)."""
        return self.dropout(x + self.pe)


class _EfficientRelativePositionAttention(nn.Module):
    """Original efficient relative position encoding (eRPE) attention."""

    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size**-0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.seq_len - 1), num_heads)
        )
        coords = torch.meshgrid(
            torch.arange(1), torch.arange(self.seq_len), indexing="ij"
        )
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        # Kept because it is part of the original module/state, although the
        # authors' eRPE forward pass does not apply it.
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        """Apply multi-head attention followed by the original eRPE bias."""
        batch_size, seq_len, _ = x.shape
        key = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        value = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        query = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )

        attention = torch.matmul(query, key) * self.scale
        attention = nn.functional.softmax(attention, dim=-1)

        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.num_heads)
        )
        relative_bias = (
            relative_bias.reshape(self.seq_len, self.seq_len, self.num_heads)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        attention = attention + relative_bias

        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.to_out(output)


class _ConvTranNetwork(nn.Module):
    """Original ConvTran network using tAPE and eRPE."""

    def __init__(
        self,
        n_channels,
        n_timepoints,
        n_classes,
        emb_size,
        num_heads,
        dim_ff,
        dropout,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, emb_size * 4, kernel_size=(1, 8), padding="same"),
            nn.BatchNorm2d(emb_size * 4),
            nn.GELU(),
        )
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(
                emb_size * 4,
                emb_size,
                kernel_size=(n_channels, 1),
                padding="valid",
            ),
            nn.BatchNorm2d(emb_size),
            nn.GELU(),
        )
        self.fix_position = _TimeAbsolutePositionalEncoding(
            emb_size, dropout=dropout, max_len=n_timepoints
        )
        self.attention_layer = _EfficientRelativePositionAttention(
            emb_size, num_heads, n_timepoints, dropout
        )
        self.layer_norm = nn.LayerNorm(emb_size, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(emb_size, eps=1e-5)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        """Return logits for aeon-shaped input (batch, channels, timepoints)."""
        if x.ndim != 3:
            raise ValueError("ConvTran expects a 3D tensor")
        if x.shape[1] != self.n_channels or x.shape[2] != self.n_timepoints:
            raise ValueError(
                "ConvTran input shape changed after fitting: expected "
                f"(*, {self.n_channels}, {self.n_timepoints}), got {tuple(x.shape)}"
            )

        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        x_src_pos = self.fix_position(x_src)
        attention = x_src + self.attention_layer(x_src_pos)
        attention = self.layer_norm(attention)
        output = attention + self.feed_forward(attention)
        output = self.layer_norm2(output)
        output = output.permute(0, 2, 1)
        output = self.gap(output)
        output = self.flatten(output)
        return self.out(output)


class _RAdam(Optimizer):
    """RAdam optimizer copied from the authors' ConvTran implementation."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if "betas" in param and param["betas"] != betas:
                    param["buffer"] = [[None, None, None] for _ in range(10)]
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "buffer": [[None, None, None] for _ in range(10)],
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one optimizer step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.float()
                if gradient.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                parameter_fp32 = parameter.float()
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(parameter_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(parameter_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(parameter_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1 - beta2
                )
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)

                state["step"] += 1
                buffered = group["buffer"][state["step"] % 10]
                if state["step"] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = (
                        n_sma_max
                        - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    )
                    buffered[1] = n_sma
                    if n_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (n_sma - 4)
                            / (n_sma_max - 4)
                            * (n_sma - 2)
                            / n_sma
                            * n_sma_max
                            / (n_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                if n_sma >= 5:
                    if group["weight_decay"] != 0:
                        parameter_fp32.add_(
                            parameter_fp32,
                            alpha=-group["weight_decay"] * group["lr"],
                        )
                    denominator = exp_avg_sq.sqrt().add_(group["eps"])
                    parameter_fp32.addcdiv_(
                        exp_avg,
                        denominator,
                        value=-step_size * group["lr"],
                    )
                    parameter.copy_(parameter_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        parameter_fp32.add_(
                            parameter_fp32,
                            alpha=-group["weight_decay"] * group["lr"],
                        )
                    parameter_fp32.add_(
                        exp_avg, alpha=-step_size * group["lr"]
                    )
                    parameter.copy_(parameter_fp32)
        return loss


class ConvTranClassifier(BaseClassifier):
    """Convolutional Transformer (ConvTran) classifier.

    This is a direct PyTorch port of the authors' ConvTran model and training
    procedure, adapted to the aeon estimator interface. Input is an aeon-format
    NumPy array with shape ``(n_cases, n_channels, n_timepoints)``. The original
    implementation uses the same layout.

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
        Fraction of the supplied training data held out for validation and
        best-loss checkpoint selection. Set to 0 to train on all supplied data
        and retain the epoch with the lowest training loss.
    gradient_clip_norm : float or None, default=4.0
        Maximum gradient norm, matching the original training loop. No clipping
        is performed when ``None``.
    device : {"auto", "cpu", "cuda"} or torch device string, default="auto"
        Device used for training and prediction. ``"auto"`` selects CUDA when
        available and otherwise CPU.
    num_workers : int, default=0
        Number of data-loader worker processes.
    verbose : bool, default=False
        Whether to print epoch losses.
    random_state : int, RandomState instance or None, default=1234
        Seed controlling the validation split, model initialization, and batch
        shuffling. The default matches the original implementation.

    Attributes
    ----------
    model_ : torch.nn.Module
        Fitted network restored to the epoch with the best validation loss.
    history_ : list of dict
        Training and validation loss for each epoch.
    device_ : torch.device
        Resolved training device.
    best_epoch_ : int
        One-based index of the retained epoch.
    best_validation_loss_ : float
        Loss used to choose the retained epoch.

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
        "cant_pickle": True,
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
            y_tensor = torch.as_tensor(y, dtype=torch.int64)
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
                logits = self.model_(X_batch)
                losses = nn.functional.cross_entropy(
                    logits, y_batch, reduction="none"
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

        encoded_y = np.asarray(
            [self._class_dictionary[label] for label in y], dtype=np.int64
        )
        if self.validation_size > 0:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.validation_size,
                random_state=self.random_state_,
            )
            train_indices, validation_indices = next(
                splitter.split(np.zeros(len(encoded_y)), encoded_y)
            )
            X_train, y_train = X[train_indices], encoded_y[train_indices]
            X_validation = X[validation_indices]
            y_validation = encoded_y[validation_indices]
        else:
            X_train, y_train = X, encoded_y
            X_validation = y_validation = None

        self.model_ = _ConvTranNetwork(
            n_channels=self.n_channels_,
            n_timepoints=self.n_timepoints_,
            n_classes=self.n_classes_,
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

    def _predict_proba(self, X):
        loader = self._make_loader(X, shuffle=False)
        self.model_.eval()
        probabilities = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                logits = self.model_(X_batch)
                probabilities.append(nn.functional.softmax(logits, dim=1).cpu())
        return torch.cat(probabilities, dim=0).numpy()

    def _predict(self, X):
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]

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
