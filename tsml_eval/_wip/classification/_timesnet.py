"""TimesNet classifier for aeon.

Adapted from the THUML Time-Series-Library implementation of TimesNet:
- models/TimesNet.py
- layers/Embed.py
- layers/Conv_Blocks.py

Original repository:
https://github.com/thuml/Time-Series-Library

This wrapper is designed for aeon and therefore assumes input X is a 3D NumPy
array with shape (n_cases, n_channels, n_timepoints). The original TimesNet
classification implementation expects tensors with shape
(batch, time, channels), so X is transposed internally before being passed to
the PyTorch network.

The original source is distributed under the MIT License.

MIT License

Copyright (c) 2021 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

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

from __future__ import annotations

__maintainer__ = ["TonyBagnall"]
__all__ = ["TimesNetClassifier"]

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier



class TimesNetClassifier(BaseClassifier):
    """
      TimesNet classifier adapted for aeon.

      This estimator is designed to stay close to the original TSLib TimesNet
      classification implementation, while conforming to aeon conventions.

      TimesNet models temporal variation by identifying dominant periods with an
      FFT-based procedure, reshaping the series into multiple 2D representations,
      and applying inception-style 2D convolutions within stacked TimesBlocks.

      Parameters
      ----------
      e_layers : int, default=2
          Number of TimesBlocks.
      d_model : int, default=32
          Embedding dimension. The default is the modal value across the
          authors' published classification scripts, which use 32 or 64.
      d_ff : int, default=64
          Hidden dimension inside the inception-style convolution blocks. The
          default is the modal value across the authors' published
          classification scripts, which use 32, 64 or 256.
      top_k : int, default=3
          Number of dominant periods selected from the FFT.
      num_kernels : int, default=6
          Number of inception kernels per block.
      dropout : float, default=0.1
          Dropout rate.
      batch_size : int, default=16
          Training batch size.
      n_epochs : int, default=30
          Maximum number of training epochs.
      learning_rate : float, default=1e-3
          Learning rate for RAdam.
      lr_adjust : {"type1", "cosine", None}, default=None
          Learning rate schedule, applied every five epochs as in the original
          TSLib training loop. ``"type1"`` is TSLib's own default and sets the
          rate to ``learning_rate * 0.5 ** (epoch - 1)`` at epochs 5, 10, 15 and
          so on, which from the published ``learning_rate=0.001`` reaches 2.0e-6
          by epoch 10 and 6.1e-8 by epoch 15, so training effectively stops
          about a third of the way through a 30 epoch run.

          The default here is None, which departs from TSLib deliberately. Their
          schedule is harmless in their pipeline because they select the
          retained epoch on the *test* set, so an early epoch from before the
          collapse is kept anyway. This wrapper selects on a held-out split of
          the training data, so it keeps a model that stopped learning. On ERing
          the difference is 0.578 with the schedule against 0.933 without, where
          the published result is 0.915. Set ``"type1"`` to reproduce TSLib's
          behaviour, or ``"cosine"`` for their cosine option.
      patience : int, default=10
          Early stopping patience based on internal validation accuracy.
      validation_size : float, default=0.2
          Proportion of the training set used for internal validation.
      gradient_clip : float, default=4.0
          Maximum gradient norm.
      standardise : bool, default=True
          Whether to apply per-channel standardisation fitted on the training data.
      device : str or None, default=None
          Torch device string. If None, chooses CUDA when available, else CPU.
      random_state : int or None, default=None
          Random seed.
      verbose : bool, default=False
          If True, print training progress.

      Attributes
      ----------
      network_ : torch.nn.Module
          Fitted network, restored to the epoch with the best validation score.
      scaler_ : object or None
          Fitted per-channel standardiser, or None when ``standardise`` is False.
      seq_len_ : int
          Series length seen in ``fit``. ``predict`` requires the same length.
      n_channels_ : int
          Number of channels seen in ``fit``.
      device_ : torch.device
          Resolved training device.
      history_ : list of dict
          One entry per epoch with ``epoch``, ``train_loss`` and
          ``validation_accuracy``. When ``validation_size`` is 0 there is no
          held-out set, and ``validation_accuracy`` holds the negated training
          loss, which is what selection then uses.
      best_epoch_ : int
          One-based index of the retained epoch.
      best_validation_score_ : float
          Score that selected the retained epoch: validation accuracy, or the
          negated training loss when there is no validation set.
      classes_ : np.ndarray
          Class labels, from ``BaseClassifier``.
      n_classes_ : int
          Number of classes, from ``BaseClassifier``.

      Notes
      -----
      - Input X must be a NumPy array of shape (n_cases, n_channels, n_timepoints).
      - Equal-length series are assumed.
      - Internally, X is transposed to (n_cases, n_timepoints, n_channels) before
        being passed to the PyTorch model because that is the layout expected by
        the original TimesNet implementation.
      - The original repository uses an explicit validation loader and early
        stopping. This wrapper reproduces that behaviour via an internal split
        from the training data.

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
        gradient_clip: float = 4.0,
        standardise: bool = True,
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
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        super().__init__()

    def _validate_parameters(self) -> None:
        """Check constructor parameters before any work is done."""
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
        if self.lr_adjust not in (None, "none", "type1", "cosine"):
            raise ValueError(
                'lr_adjust must be one of "type1", "cosine", or None, got '
                f"{self.lr_adjust!r}"
            )

    def _resolve_device(self) -> torch.device:
        """Resolve torch device, rejecting an unavailable CUDA request."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(self.device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved

    @staticmethod
    def _preprocess_X(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert aeon numpy3D input to TimesNet layout and create a mask.

        Parameters
        ----------
        X : np.ndarray, shape (n_cases, n_channels, n_timepoints)

        Returns
        -------
        x_t : np.ndarray, shape (n_cases, n_timepoints, n_channels)
            Transposed input for the TimesNet network.
        mask : np.ndarray, shape (n_cases, n_timepoints)
            All-ones mask. It is retained because the original classification
            implementation multiplies the network output by a padding mask.
        """
        x_t = np.transpose(X, (0, 2, 1)).astype(np.float32, copy=False)
        mask = np.ones((X.shape[0], X.shape[2]), dtype=np.float32)
        return x_t, mask

    def _encode_y(self, y) -> np.ndarray:
        """Map class labels to integer indices."""
        return np.asarray([self._class_dictionary[label] for label in y], dtype=np.int64)

    def _make_loader(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray | None = None,
        shuffle: bool = False,
    ) -> torch.utils.data.DataLoader:
        """Create a PyTorch DataLoader."""
        x_tensor = torch.from_numpy(x)
        mask_tensor = torch.from_numpy(mask)

        if y is None:
            dataset = torch.utils.data.TensorDataset(x_tensor, mask_tensor)
        else:
            y_tensor = torch.from_numpy(y)
            dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, mask_tensor)

        generator = None
        if self.random_state is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.random_state))

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def _build_model(self, n_channels: int) -> _TimesNetClassificationModel:
        """Construct the underlying TimesNet model."""
        return _TimesNetClassificationModel(
            seq_len=self.seq_len_,
            enc_in=n_channels,
            num_class=self.n_classes_,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            top_k=self.top_k,
            num_kernels=self.num_kernels,
            dropout=self.dropout,
        )

    def _score_loader(self, loader: torch.utils.data.DataLoader) -> float:
        """Compute classification accuracy for a loader."""
        self.network_.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for batch_x, batch_y, batch_mask in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                batch_mask = batch_mask.to(self.device_)

                logits = self.network_(batch_x, batch_mask)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                trues.append(batch_y.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        return float(np.mean(preds == trues))

    def _adjust_learning_rate(self, optimiser, epoch: int) -> float | None:
        """Apply the TSLib learning rate schedule for a one-based ``epoch``.

        Mirrors ``utils/tools.py::adjust_learning_rate`` in the original
        Time-Series-Library, which the classification training loop calls once
        every five epochs. Returns the new rate, or None if unchanged.
        """
        if self.lr_adjust in (None, "none") or epoch % 5 != 0:
            return None

        if self.lr_adjust == "type1":
            lr = self.learning_rate * (0.5 ** (epoch - 1))
        else:  # cosine
            lr = (
                self.learning_rate
                / 2
                * (1 + math.cos(epoch / self.n_epochs * math.pi))
            )

        for param_group in optimiser.param_groups:
            param_group["lr"] = lr
        return lr

    def _fit(self, X: np.ndarray, y):
        """
        Fit TimesNet.

        X must have shape (n_cases, n_channels, n_timepoints).
        """
        self._validate_parameters()
        _set_torch_seed(self.random_state)
        rng = check_random_state(self.random_state)

        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError(
                "TimesNetClassifier requires X to be a 3D NumPy array with "
                "shape (n_cases, n_channels, n_timepoints)."
            )

        self.n_channels_ = X.shape[1]
        self.seq_len_ = X.shape[2]

        x_t, mask = self._preprocess_X(X)

        if self.standardise:
            self.scaler_ = _StandardisePerChannel().fit(x_t)
            x_t = self.scaler_.transform(x_t)
        else:
            self.scaler_ = None

        y_int = self._encode_y(y)

        if self.validation_size and 0 < self.validation_size < 1 and len(x_t) >= 2:
            bincounts = np.bincount(y_int)
            stratify = y_int if np.min(bincounts) >= 2 else None
            train_idx, val_idx = train_test_split(
                np.arange(len(x_t)),
                test_size=self.validation_size,
                random_state=rng.randint(np.iinfo(np.int32).max),
                stratify=stratify,
            )
        else:
            train_idx = np.arange(len(x_t))
            val_idx = np.array([], dtype=int)

        self.network_ = self._build_model(self.n_channels_)
        self.device_ = self._resolve_device()
        self.network_.to(self.device_)

        optimiser = torch.optim.RAdam(
            self.network_.parameters(), lr=self.learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        train_loader = self._make_loader(
            x_t[train_idx], mask[train_idx], y_int[train_idx], shuffle=True
        )
        val_loader = (
            None
            if len(val_idx) == 0
            else self._make_loader(
                x_t[val_idx], mask[val_idx], y_int[val_idx], shuffle=False
            )
        )

        best_state = deepcopy(self.network_.state_dict())
        best_score = -np.inf
        epochs_without_improvement = 0
        self.history_ = []

        for epoch in range(self.n_epochs):
            self.network_.train()
            train_loss = 0.0
            n_train = 0

            for batch_x, batch_y, batch_mask in train_loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                batch_mask = batch_mask.to(self.device_)

                optimiser.zero_grad()
                logits = self.network_(batch_x, batch_mask)
                loss = criterion(logits, batch_y)
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.network_.parameters(), max_norm=self.gradient_clip
                )
                optimiser.step()

                bs = batch_x.shape[0]
                train_loss += loss.item() * bs
                n_train += bs

            train_loss /= max(n_train, 1)

            if val_loader is None:
                score = -train_loss
            else:
                score = self._score_loader(val_loader)

            self.history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_accuracy": score,
                }
            )

            if self.verbose:
                if val_loader is None:
                    print(
                        f"epoch={epoch + 1} "
                        f"train_loss={train_loss:.6f}"
                    )
                else:
                    print(
                        f"epoch={epoch + 1} "
                        f"train_loss={train_loss:.6f} "
                        f"val_acc={score:.6f}"
                    )

            if score > best_score:
                best_score = score
                best_state = deepcopy(self.network_.state_dict())
                self.best_epoch_ = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if val_loader is not None and epochs_without_improvement >= self.patience:
                    break

            new_lr = self._adjust_learning_rate(optimiser, epoch + 1)
            if new_lr is not None and self.verbose:
                print(f"updating learning rate to {new_lr}")

        self.network_.load_state_dict(best_state)
        self.best_validation_score_ = best_score
        return self

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        X must have shape (n_cases, n_channels, n_timepoints).
        """
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError(
                "TimesNetClassifier requires X to be a 3D NumPy array with "
                "shape (n_cases, n_channels, n_timepoints)."
            )

        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"X has {X.shape[1]} channels, but classifier was fitted with "
                f"{self.n_channels_} channels."
            )

        if X.shape[2] != self.seq_len_:
            raise ValueError(
                f"X has length {X.shape[2]}, but classifier was fitted with "
                f"seq_len={self.seq_len_}."
            )

        x_t, mask = self._preprocess_X(X)

        if self.scaler_ is not None:
            x_t = self.scaler_.transform(x_t)

        loader = self._make_loader(x_t, mask, y=None, shuffle=False)

        self.network_.eval()
        probs = []

        with torch.no_grad():
            for batch_x, batch_mask in loader:
                batch_x = batch_x.to(self.device_)
                batch_mask = batch_mask.to(self.device_)

                logits = self.network_(batch_x, batch_mask)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        return np.concatenate(probs, axis=0)

    def _predict(self, X: np.ndarray):
        """Predict class labels."""
        probs = self._predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return self.classes_[preds]

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict:
        """Return testing parameter settings."""
        return {
            "e_layers": 1,
            "d_model": 16,
            "d_ff": 16,
            "top_k": 2,
            "num_kernels": 2,
            "dropout": 0.1,
            "batch_size": 4,
            "n_epochs": 2,
            "learning_rate": 1e-3,
            "patience": 2,
            "validation_size": 0.2,
            "gradient_clip": 4.0,
            "standardise": True,
            "random_state": 0,
            "verbose": False,
        }

def _set_torch_seed(random_state: int | None) -> None:
    """Set PyTorch random seeds."""
    if random_state is None:
        return

    seed = int(random_state)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _PositionalEmbedding(nn.Module):
    """Positional embedding adapted from TSLib layers/Embed.py."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional encodings for the current sequence length."""
        return self.pe[:, : x.size(1)]


class _TokenEmbedding(nn.Module):
    """Token embedding adapted from TSLib layers/Embed.py."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()

        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed channels into the model dimension."""
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class _DataEmbedding(nn.Module):
    """
    Data embedding adapted from TSLib layers/Embed.py.

    In the upstream classification code, TimesNet calls DataEmbedding with no
    temporal marks, so this aeon wrapper keeps value and positional embeddings
    only.
    """

    def __init__(
        self, c_in: int, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.value_embedding = _TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = _PositionalEmbedding(
            d_model=d_model, max_len=max_len
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply value and positional embeddings."""
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class _InceptionBlockV1(nn.Module):
    """Inception block adapted from TSLib layers/Conv_Blocks.py."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        init_weight: bool = True,
    ):
        super().__init__()

        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * i + 1,
                    padding=i,
                )
            )
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialise convolution kernels."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all kernel sizes and average their outputs."""
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        return torch.stack(res_list, dim=-1).mean(-1)


def _fft_for_period(x: torch.Tensor, k: int = 2) -> tuple[np.ndarray, torch.Tensor]:
    """
    FFT-based period selection adapted from TSLib models/TimesNet.py.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape (batch, time, channels), matching the upstream
        TimesNet implementation.
    k : int, default=2
        Number of periods to select.

    Returns
    -------
    period : np.ndarray
        Selected periods as integer divisors derived from Fourier frequencies.
    period_weight : torch.Tensor
        Batch-wise weights for the selected frequencies.
    """
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = torch.abs(xf).mean(0).mean(-1)

    # Avoid selecting the zero-frequency component.
    if frequency_list.shape[0] > 0:
        frequency_list = frequency_list.clone()
        frequency_list[0] = 0

    # Guard against very short series.
    k = max(1, min(k, frequency_list.shape[0]))
    _, top_list = torch.topk(frequency_list, k)

    # A zero index would imply division by zero. Replace it with 1 safely.
    top_list = torch.where(
        top_list == 0,
        torch.ones_like(top_list),
        top_list,
    )

    period = x.shape[1] // top_list.detach().cpu().numpy()
    period = np.maximum(period, 1)

    period_weight = torch.abs(xf).mean(-1)[:, top_list]
    return period, period_weight


class _TimesBlock(nn.Module):
    """TimesBlock adapted from TSLib models/TimesNet.py."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int,
        d_model: int,
        d_ff: int,
        num_kernels: int,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k

        self.conv = nn.Sequential(
            _InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            _InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply TimesNet temporal block."""
        b, t, n = x.size()
        period_list, period_weight = _fft_for_period(x, self.k)

        total_len = self.seq_len + self.pred_len
        res = []

        for i in range(len(period_list)):
            period = int(period_list[i])

            if total_len % period != 0:
                length = ((total_len // period) + 1) * period
                padding = torch.zeros(
                    (x.shape[0], length - total_len, x.shape[2]),
                    device=x.device,
                    dtype=x.dtype,
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x

            out = (
                out.reshape(b, length // period, period, n)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(b, -1, n)
            res.append(out[:, :total_len, :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = (
            period_weight.unsqueeze(1).unsqueeze(1).repeat(1, t, n, 1)
        )
        res = torch.sum(res * period_weight, dim=-1)

        return res + x


class _TimesNetClassificationModel(nn.Module):
    """
    Classification-only TimesNet model adapted from TSLib models/TimesNet.py.

    This follows the upstream classification path:
    embedding -> stacked TimesBlocks + layer norm -> GELU -> dropout ->
    flatten -> linear projection.
    """

    def __init__(
        self,
        seq_len: int,
        enc_in: int,
        num_class: int,
        e_layers: int = 2,
        d_model: int = 32,
        d_ff: int = 64,
        top_k: int = 3,
        num_kernels: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.layer = e_layers

        self.model = nn.ModuleList(
            [
                _TimesBlock(
                    seq_len=seq_len,
                    pred_len=0,
                    top_k=top_k,
                    d_model=d_model,
                    d_ff=d_ff,
                    num_kernels=num_kernels,
                )
                for _ in range(e_layers)
            ]
        )

        # The positional table must cover the series. TSLib fixes it at 5000,
        # which fails on the longer Multiverse problems: EigenWorms is 17984
        # points and Alzheimers 15000.
        self.enc_embedding = _DataEmbedding(
            enc_in, d_model, dropout=dropout, max_len=max(5000, seq_len)
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * seq_len, num_class)

    def forward(self, x_enc: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x_enc : torch.Tensor
            Shape (batch, time, channels), matching the upstream TimesNet code.
        padding_mask : torch.Tensor
            Shape (batch, time). In this aeon wrapper this is all ones because
            equal-length series are assumed.
        """
        enc_out = self.enc_embedding(x_enc)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)

        # Retained for faithfulness to the original classification branch.
        output = output * padding_mask.unsqueeze(-1)

        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output


class _StandardisePerChannel:
    """
    Per-channel standardisation over training data.

    Statistics are computed across all cases and time points for each channel.
    Input is assumed to have shape (n_cases, n_timepoints, n_channels).
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "_StandardisePerChannel":
        """Estimate per-channel mean and standard deviation."""
        self.mean_ = x.mean(axis=(0, 1)).astype(np.float32)
        self.std_ = x.std(axis=(0, 1)).astype(np.float32)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply standardisation using fitted statistics."""
        return ((x - self.mean_[None, None, :]) / self.std_[None, None, :]).astype(
            np.float32
        )
