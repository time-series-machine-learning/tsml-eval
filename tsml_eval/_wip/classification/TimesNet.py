"""
TimesNet classifier (PyTorch).

Ported for aeon from the TimesNet classification pathway in THUML's Time-Series-Library
(MIT License). The original repo pads unequal-length series to seq_len and provides a
padding mask to the model; the encoder output is multiplied by this mask prior to
flattening and classification.

Notes
-----
- This is a PyTorch implementation. It does not depend on TensorFlow/Keras.
- Unequal length input is supported via zero-padding + padding_mask.
- At predict-time, series longer than the fitted seq_len_ raise an error because the
  final linear layer depends on seq_len_.

References
----------
Wu, H. et al. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.
ICLR 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.utils.validation._dependencies import _check_soft_dependencies


class TimesNetClassifier(BaseClassifier):
    """TimesNet classifier (PyTorch).

    Parameters
    ----------
    seq_len : int or None, default=None
        Length to pad/truncate to during fit. If None, uses max length seen in X during fit.
        If provided, any training series longer than seq_len raises ValueError.
    top_k : int, default=3
        Number of dominant periods selected from FFT.
    e_layers : int, default=2
        Number of TimesBlocks.
    d_model : int, default=64
        Embedding dimension.
    d_ff : int, default=128
        Hidden dimension inside the 2D inception-style block.
    num_kernels : int, default=6
        Number of kernels (scales) in the inception-style conv block.
    dropout : float, default=0.1
        Dropout probability.
    batch_size : int, default=32
        Mini-batch size.
    n_epochs : int, default=50
        Number of epochs.
    learning_rate : float, default=1e-3
        Optimiser learning rate.
    weight_decay : float, default=0.0
        Weight decay (L2).
    optimiser : {"adam", "radam"}, default="adam"
        Optimiser choice. "radam" is used in the upstream repo for classification runs,
        but may not exist in some torch builds; if unavailable, it falls back to Adam.
    clip_grad_norm : float or None, default=4.0
        Max norm for gradient clipping. If None, disables clipping.
    val_split : float, default=0.0
        Fraction of training set to use for validation-based early stopping. If 0, no val split.
    patience : int, default=10
        Early stopping patience (only used if val_split > 0).
    device : {"auto", "cpu", "cuda"}, default="auto"
        Torch device selection.
    random_state : int, RandomState instance or None, default=None
        RNG seed for shuffling and initialisation.
    verbose : bool, default=False
        Print training progress.

    Notes
    -----
    - Tags set python_dependencies="torch" so aeon can soft-skip if torch not installed.
    """

    _tags = {
        "python_dependencies": "torch",
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "cant_pickle": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": False,
        "X_inner_type": ["numpy3D", "np-list"],
    }

    def __init__(
        self,
        seq_len: Optional[int] = None,
        top_k: int = 3,
        e_layers: int = 2,
        d_model: int = 64,
        d_ff: int = 128,
        num_kernels: int = 6,
        dropout: float = 0.1,
        batch_size: int = 32,
        n_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimiser: str = "adam",
        clip_grad_norm: Optional[float] = 4.0,
        val_split: float = 0.0,
        patience: int = 10,
        device: str = "auto",
        random_state=None,
        verbose: bool = False,
    ):
        self.seq_len = seq_len
        self.top_k = top_k
        self.e_layers = e_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.dropout = dropout

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimiser = optimiser
        self.clip_grad_norm = clip_grad_norm

        self.val_split = val_split
        self.patience = patience

        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        # fitted
        self.model_ = None
        self.seq_len_ = None
        self.n_channels_ = None
        self.device_ = None

        super().__init__()

    @staticmethod
    def _import_torch():
        _check_soft_dependencies("torch", severity="error")
        import torch  # noqa: WPS433

        return torch

    def _select_device(self, torch):
        if self.device == "cpu":
            return torch.device("cpu")
        if self.device == "cuda":
            return torch.device("cuda")
        # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _pad_and_mask(
        self, X, *, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_pad[B,L,C], mask[B,L], lengths[B])."""
        if isinstance(X, np.ndarray):
            # X is numpy3D: (B, C, T)
            B, C, T = X.shape
            if T > seq_len:
                raise ValueError(
                    f"Found series length {T} > seq_len={seq_len}. "
                    "Set TimesNetClassifier(seq_len=...) large enough."
                )
            X_pad = np.zeros((B, seq_len, C), dtype=np.float32)
            X_pad[:, :T, :] = X.transpose(0, 2, 1).astype(np.float32)
            mask = np.zeros((B, seq_len), dtype=np.float32)
            mask[:, :T] = 1.0
            lengths = np.full((B,), T, dtype=np.int64)
            return X_pad, mask, lengths

        # X is np-list: list of arrays (C, Ti)
        lengths = np.array([x.shape[1] for x in X], dtype=np.int64)
        C = X[0].shape[0]
        if np.any(lengths > seq_len):
            mx = int(lengths.max())
            raise ValueError(
                f"Found series length {mx} > fitted seq_len={seq_len}. "
                "TimesNet uses a fixed-length flatten head, so it cannot "
                "predict on longer series than seen/declared at fit-time."
            )

        B = len(X)
        X_pad = np.zeros((B, seq_len, C), dtype=np.float32)
        mask = np.zeros((B, seq_len), dtype=np.float32)
        for i, x in enumerate(X):
            Ti = x.shape[1]
            X_pad[i, :Ti, :] = x.T.astype(np.float32)
            mask[i, :Ti] = 1.0
        return X_pad, mask, lengths

    def _build_model(self, *, seq_len: int, n_channels: int, n_classes: int):
        torch = self._import_torch()
        import math  # noqa: WPS433

        nn = torch.nn
        F = torch.nn.functional

        class PositionalEmbedding(nn.Module):
            def __init__(self, d_model: int, max_len: int = 5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

            def forward(self, x):
                # x: [B, L, D]
                return self.pe[:, : x.size(1)].to(dtype=x.dtype)

        class TokenEmbedding(nn.Module):
            def __init__(self, c_in: int, d_model: int):
                super().__init__()
                self.tokenConv = nn.Conv1d(
                    in_channels=c_in,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    padding_mode="circular",
                    bias=False,
                )
                nn.init.kaiming_normal_(
                    self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

            def forward(self, x):
                # x: [B, L, C] -> [B, C, L] -> conv -> [B, L, D]
                x = x.permute(0, 2, 1)
                x = self.tokenConv(x)
                return x.transpose(1, 2)

        class DataEmbedding(nn.Module):
            def __init__(self, c_in: int, d_model: int, dropout: float):
                super().__init__()
                self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
                self.position_embedding = PositionalEmbedding(d_model=d_model)
                self.dropout = nn.Dropout(p=dropout)

            def forward(self, x):
                x = self.value_embedding(x) + self.position_embedding(x)
                return self.dropout(x)

        class InceptionBlockV1(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, num_kernels: int):
                super().__init__()
                self.num_kernels = num_kernels
                self.kernels = nn.ModuleList()
                for i in range(num_kernels):
                    k = 2 * i + 1
                    p = i
                    self.kernels.append(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=(k, k),
                            padding=(p, p),
                            bias=True,
                        )
                    )

            def forward(self, x):
                # x: [B, C, H, W]
                outs = [conv(x) for conv in self.kernels]
                out = torch.stack(outs, dim=0).mean(dim=0)
                return out

        def FFT_for_Period(x, k: int):
            # x: [B, L, D]
            xf = torch.fft.rfft(x, dim=1)  # [B, F, D]
            amp = torch.abs(xf).mean(0).mean(-1)  # [F]
            if amp.numel() > 0:
                amp[0] = 0.0
            top = torch.topk(amp, k=min(k, amp.numel()), dim=0).indices
            top = torch.clamp(top, min=1)  # avoid divide by 0
            period = (x.shape[1] // top).to(torch.long)
            period = torch.clamp(period, min=1)
            weight = torch.abs(xf).mean(-1)[:, top]  # [B, k]
            return period, weight

        class TimesBlock(nn.Module):
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
                self.top_k = top_k
                self.conv = nn.Sequential(
                    InceptionBlockV1(d_model, d_ff, num_kernels),
                    nn.GELU(),
                    InceptionBlockV1(d_ff, d_model, num_kernels),
                )

            def forward(self, x):
                # x: [B, L, D]
                B, L, Dm = x.shape
                periods, period_weight = FFT_for_Period(x, self.top_k)  # [k], [B,k]
                res = []
                base_len = self.seq_len + self.pred_len

                for i in range(periods.numel()):
                    p = int(periods[i].item())
                    if p <= 0:
                        p = 1

                    if base_len % p != 0:
                        length = ((base_len // p) + 1) * p
                        pad_len = length - base_len
                        pad = torch.zeros((B, pad_len, Dm), device=x.device, dtype=x.dtype)
                        x_pad = torch.cat([x[:, :base_len, :], pad], dim=1)
                    else:
                        length = base_len
                        x_pad = x[:, :base_len, :]

                    # [B, length, D] -> [B, D, length//p, p]
                    out = x_pad.reshape(B, length // p, p, Dm).permute(0, 3, 1, 2)
                    out = self.conv(out)
                    # back to [B, length, D]
                    out = out.permute(0, 2, 3, 1).reshape(B, length, Dm)
                    out = out[:, :base_len, :]
                    res.append(out)

                res = torch.stack(res, dim=-1)  # [B, base_len, D, k]
                w = torch.softmax(period_weight, dim=1)  # [B, k]
                w = w.unsqueeze(1).unsqueeze(2)  # [B,1,1,k]
                out = (res * w).sum(dim=-1)  # [B, base_len, D]
                return out + x[:, :base_len, :]

        class TimesNetModel(nn.Module):
            def __init__(
                self,
                seq_len: int,
                n_channels: int,
                n_classes: int,
                top_k: int,
                e_layers: int,
                d_model: int,
                d_ff: int,
                num_kernels: int,
                dropout: float,
            ):
                super().__init__()
                self.seq_len = seq_len
                self.pred_len = 0

                self.enc_embedding = DataEmbedding(
                    c_in=n_channels, d_model=d_model, dropout=dropout
                )
                self.model = nn.ModuleList(
                    [
                        TimesBlock(
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
                self.layer_norm = nn.LayerNorm(d_model)
                self.act = nn.GELU()
                self.drop = nn.Dropout(p=dropout)

                self.projection = nn.Linear(d_model * seq_len, n_classes)

            def forward(self, x_enc, padding_mask):
                # x_enc: [B, L, C], padding_mask: [B, L]
                enc_out = self.enc_embedding(x_enc)

                for block in self.model:
                    enc_out = block(enc_out)
                    enc_out = self.layer_norm(enc_out)

                enc_out = self.act(enc_out)
                enc_out = self.drop(enc_out)

                if padding_mask is not None:
                    enc_out = enc_out * padding_mask.unsqueeze(-1)

                enc_out = enc_out.reshape(enc_out.size(0), -1)  # [B, L*D]
                return self.projection(enc_out)

        return TimesNetModel(
            seq_len=seq_len,
            n_channels=n_channels,
            n_classes=n_classes,
            top_k=self.top_k,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_kernels=self.num_kernels,
            dropout=self.dropout,
        )

    def _fit(self, X, y):
        torch = self._import_torch()
        nn = torch.nn

        rng = check_random_state(self.random_state)
        seed = int(rng.randint(0, np.iinfo(np.int32).max))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # derive seq_len_ from training data unless explicitly set
        if self.seq_len is None:
            if isinstance(X, np.ndarray):
                seq_len = X.shape[2]
            else:
                seq_len = int(max(x.shape[1] for x in X))
        else:
            seq_len = int(self.seq_len)

        # pad + mask, and store fitted shape
        X_pad, mask, _ = self._pad_and_mask(X, seq_len=seq_len)
        n_cases, seq_len2, n_channels = X_pad.shape
        if seq_len2 != seq_len:
            raise RuntimeError("Internal padding produced unexpected seq_len.")

        self.seq_len_ = seq_len
        self.n_channels_ = n_channels

        # encode y to integers using aeon mapping
        y_int = np.array([self._class_dictionary[yy] for yy in y], dtype=np.int64)

        # train/val split
        idx = np.arange(n_cases)
        rng.shuffle(idx)
        if self.val_split and self.val_split > 0.0:
            n_val = max(1, int(round(n_cases * float(self.val_split))))
            val_idx = idx[:n_val]
            tr_idx = idx[n_val:]
        else:
            val_idx = None
            tr_idx = idx

        def make_loader(indices, shuffle: bool):
            from torch.utils.data import DataLoader, TensorDataset  # noqa: WPS433

            xb = torch.from_numpy(X_pad[indices]).float()
            mb = torch.from_numpy(mask[indices]).float()
            yb = torch.from_numpy(y_int[indices]).long()
            ds = TensorDataset(xb, mb, yb)
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=False,
            )

        train_loader = make_loader(tr_idx, shuffle=True)
        val_loader = make_loader(val_idx, shuffle=False) if val_idx is not None else None

        device = self._select_device(torch)
        self.device_ = device

        model = self._build_model(
            seq_len=self.seq_len_,
            n_channels=self.n_channels_,
            n_classes=self.n_classes_,
        ).to(device)

        # optimiser
        opt_name = str(self.optimiser).lower()
        if opt_name == "radam" and hasattr(torch.optim, "RAdam"):
            optimiser = torch.optim.RAdam(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            optimiser = torch.optim.Adam(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

        criterion = nn.CrossEntropyLoss()

        best_state = None
        best_val = -np.inf
        bad_epochs = 0

        def evaluate(loader):
            model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, mb, yb in loader:
                    xb = xb.to(device)
                    mb = mb.to(device)
                    yb = yb.to(device)
                    logits = model(xb, mb)
                    loss = criterion(logits, yb)
                    total_loss += float(loss.item()) * yb.size(0)
                    pred = torch.argmax(logits, dim=1)
                    correct += int((pred == yb).sum().item())
                    total += int(yb.size(0))
            return total_loss / max(1, total), correct / max(1, total)

        for epoch in range(int(self.n_epochs)):
            model.train()
            running = 0.0
            seen = 0

            for xb, mb, yb in train_loader:
                xb = xb.to(device)
                mb = mb.to(device)
                yb = yb.to(device)

                optimiser.zero_grad(set_to_none=True)
                logits = model(xb, mb)
                loss = criterion(logits, yb)
                loss.backward()

                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=float(self.clip_grad_norm)
                    )

                optimiser.step()

                running += float(loss.item()) * yb.size(0)
                seen += int(yb.size(0))

            train_loss = running / max(1, seen)

            if val_loader is not None:
                val_loss, val_acc = evaluate(val_loader)
                if self.verbose:
                    print(
                        f"epoch {epoch+1}/{self.n_epochs} "
                        f"train_loss={train_loss:.4f} "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                    )

                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(self.patience):
                        if self.verbose:
                            print("early stopping")
                        break
            else:
                if self.verbose:
                    print(f"epoch {epoch+1}/{self.n_epochs} train_loss={train_loss:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model_ = model
        return self

    def _predict_proba(self, X):
        torch = self._import_torch()
        F = torch.nn.functional

        X_pad, mask, _ = self._pad_and_mask(X, seq_len=int(self.seq_len_))
        device = self.device_

        xb = torch.from_numpy(X_pad).float().to(device)
        mb = torch.from_numpy(mask).float().to(device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(xb, mb)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()

        return probs

    def _predict(self, X):
        probs = self._predict_proba(X)
        idx = probs.argmax(axis=1)
        return self.classes_[idx]

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        # keep tiny for estimator checks
        return [
            {
                "n_epochs": 2,
                "batch_size": 4,
                "d_model": 16,
                "d_ff": 32,
                "e_layers": 1,
                "top_k": 2,
                "num_kernels": 3,
                "val_split": 0.0,
                "device": "cpu",
            }
        ]
