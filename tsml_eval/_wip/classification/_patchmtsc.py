"""Direct aeon wrapper for the authors' PatchMTSC implementation."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
from aeon.classification import BaseClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_random_state
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ._patchmtsc_original.model import PatchMTSC

__all__ = ["PatchMTSCClassifier"]


class PatchMTSCClassifier(BaseClassifier):
    """Original PatchMTSC network adapted to aeon's classifier interface."""

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "cant_pickle": True,
        "python_dependencies": "torch",
    }

    def __init__(self, emb_size=16, d_model_patch=16, dim_ff=256,
                 num_heads=8, patch_len=16, stride=8, dropout=0.01,
                 n_epochs=100, batch_size=16, learning_rate=1e-3,
                 validation_size=0.2, device="auto", verbose=False,
                 random_state=1234):
        self.emb_size = emb_size; self.d_model_patch = d_model_patch
        self.dim_ff = dim_ff; self.num_heads = num_heads
        self.patch_len = patch_len; self.stride = stride; self.dropout = dropout
        self.n_epochs = n_epochs; self.batch_size = batch_size
        self.learning_rate = learning_rate; self.validation_size = validation_size
        self.device = device; self.verbose = verbose; self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        if X.shape[2] < self.patch_len:
            raise ValueError("patch_len must not exceed the number of timepoints")
        if self.emb_size % 4 or self.d_model_patch % self.num_heads:
            raise ValueError("emb_size must be divisible by 4 and d_model_patch by num_heads")
        rng = check_random_state(self.random_state)
        self.random_state_ = int(rng.randint(np.iinfo(np.int32).max))
        torch.manual_seed(self.random_state_)
        self.device_ = torch.device("cuda" if self.device == "auto" and torch.cuda.is_available() else self.device)
        if self.device_.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.n_channels_, self.n_timepoints_ = X.shape[1:]
        yy = np.asarray([self._class_dictionary[v] for v in y], dtype=np.int64)
        if self.validation_size:
            tr, va = next(StratifiedShuffleSplit(1, test_size=self.validation_size,
                                                  random_state=self.random_state_).split(X, yy))
            Xtr, ytr, Xva, yva = X[tr], yy[tr], X[va], yy[va]
        else: Xtr, ytr, Xva, yva = X, yy, None, None
        cfg = {"Data_shape": X.shape, "emb_size": self.emb_size, "num_heads": self.num_heads,
               "dim_ff": self.dim_ff, "Fix_pos_encode": "tAPE", "Rel_pos_encode": "eRPE",
               "patch_len": self.patch_len, "stride": self.stride, "padding_patch": "end",
               "d_model_patch": self.d_model_patch, "d_model": self.d_model_patch,
               "dropout": self.dropout, "enc_in": self.n_channels_, "individual": 0,
               "head_dropout": 0., "pap_dropout": 0., "weight_decay": 5e-4,
               "moving_window": [2, 2], "graph_stride": [1, 2], "pool_choice": "mean"}
        self.model_ = PatchMTSC(cfg, self.n_classes_).to(self.device_)
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        def loader(a, b, shuffle):
            return DataLoader(TensorDataset(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.long)),
                              batch_size=self.batch_size, shuffle=shuffle)
        train_loader = loader(Xtr, ytr, True); val_loader = None if Xva is None else loader(Xva, yva, False)
        best, best_loss = None, float("inf"); self.history_ = []
        for epoch in range(self.n_epochs):
            self.model_.train(); total = 0.
            for xb, yb in train_loader:
                loss = nn.functional.cross_entropy(self.model_(xb.to(self.device_)), yb.to(self.device_))
                self.optimizer_.zero_grad(); loss.backward(); self.optimizer_.step(); total += loss.item()
            self.model_.eval(); losses = []
            with torch.no_grad():
                for xb, yb in (val_loader or train_loader):
                    losses.append(nn.functional.cross_entropy(self.model_(xb.to(self.device_)), yb.to(self.device_)).item())
            vl = float(np.mean(losses)); self.history_.append({"epoch": epoch + 1, "train_loss": total / len(train_loader), "validation_loss": vl})
            if vl < best_loss: best_loss, best = vl, deepcopy(self.model_.state_dict()); self.best_epoch_ = epoch + 1
            if self.verbose: print(f"Epoch {epoch + 1}/{self.n_epochs}: loss={total / len(train_loader):.6f}, val_loss={vl:.6f}")
        self.model_.load_state_dict(best); self.model_.eval(); self.best_validation_loss_ = best_loss
        return self

    def _predict_proba(self, X):
        out = []; loader = DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32)), batch_size=self.batch_size)
        with torch.no_grad():
            for (xb,) in loader: out.append(nn.functional.softmax(self.model_(xb.to(self.device_)), 1).cpu())
        return torch.cat(out).numpy()

    def _predict(self, X):
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"emb_size": 8, "d_model_patch": 8, "dim_ff": 16, "num_heads": 2,
                "patch_len": 4, "stride": 2, "n_epochs": 1, "batch_size": 4,
                "validation_size": 0.25, "device": "cpu", "random_state": 0}
