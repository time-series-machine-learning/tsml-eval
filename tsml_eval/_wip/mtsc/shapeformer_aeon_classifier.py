"""aeon-compatible wrapper for the ShapeFormer reference implementation.

Assumptions
- X passed to fit, predict, and predict_proba is a 3D numpy array with shape
  (n_cases, n_channels, n_timepoints).

Design
- No test leakage: this wrapper never uses any external test split.
- No internal validation: training runs for a fixed number of epochs.
- Shapelet discovery is run on the training data passed to fit only.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Tuple

import numpy as np

from aeon.classification.base import BaseClassifier


def _check_X3(X: Any) -> np.ndarray:
    """Validate and coerce X to float32 3D numpy.

    This wrapper assumes 3D numpy input, so we only do minimal checks.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray of shape (n_cases, n_channels, n_timepoints)")
    if X.ndim != 3:
        raise ValueError(f"X must be 3D, got ndim={X.ndim} and shape={getattr(X, 'shape', None)}")
    return X.astype(np.float32, copy=False)


def _fill_nans(X: np.ndarray, strategy: str, rng: np.random.Generator) -> np.ndarray:
    """Replace NaNs in X.

    strategy:
      - 'zero': replace NaNs with 0
      - 'noise': replace NaNs with tiny uniform noise in [0, 1e-3)
    """
    if strategy not in {"zero", "noise"}:
        raise ValueError("nan_strategy must be 'zero' or 'noise'")
    if not np.isnan(X).any():
        return X
    X2 = X.copy()
    mask = np.isnan(X2)
    if strategy == "zero":
        X2[mask] = 0.0
    else:
        X2[mask] = rng.random(np.count_nonzero(mask), dtype=np.float32) / 1000.0
    return X2


def _compute_mean_std(train_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Match the repo's mean/std computation (per channel)."""
    m_len = np.mean(train_X, axis=2)  # (n_cases, n_channels)
    mean = np.mean(m_len, axis=0)  # (n_channels,)

    s_len = np.std(train_X, axis=2)
    std = np.max(s_len, axis=0)
    std = np.where(std == 0, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def _apply_mean_std(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean[:, None]) / std[:, None]


def _patch_shapeblock_forward_for_cpu() -> None:
    """Patch ShapeBlock.forward to avoid an unconditional .cuda() call."""
    import torch

    from Models.shapeformer import ShapeBlock

    def _forward(self: Any, x: torch.Tensor):  # type: ignore
        pis = x[:, self.dim, self.start_position : self.end_position]
        ci_pis = torch.square(torch.subtract(pis[:, 1:], pis[:, :-1]))

        pis_u = pis.unfold(1, self.kernel_size, 1).contiguous()
        pis_u = pis_u.view(-1, self.kernel_size)

        ci_u = ci_pis.unfold(1, self.kernel_size - 1, 1).contiguous()
        ci_u = ci_u.view(-1, self.kernel_size - 1)
        ci_u = torch.sum(ci_u, dim=1) + (1.0 / self.norm)

        ci_shapelet_vec = torch.ones(ci_u.size(0), device=x.device, requires_grad=False) * self.ci_shapelet
        max_ci = torch.max(ci_u, ci_shapelet_vec)
        min_ci = torch.min(ci_u, ci_shapelet_vec)
        ci_dist = max_ci / min_ci
        ci_dist = torch.clamp(ci_dist, max=self.max_ci)

        dist1 = torch.sum(torch.square(pis_u - self.shapelet), 1)
        dist1 = dist1 * ci_dist
        dist1 = dist1 / self.shapelet.size(-1)
        dist1 = dist1.view(x.size(0), -1)

        index = torch.argmin(dist1, dim=1)
        pis_u = pis_u.view(x.size(0), -1, self.kernel_size)

        batch_idx = torch.arange(int(x.size(0)), device=x.device, dtype=torch.long)
        out = pis_u[batch_idx, index.to(torch.long)]

        out = self.l1(out)
        out_s = self.l2(self.shapelet.unsqueeze(0))
        out = out - out_s

        return out.view(x.shape[0], 1, -1)

    ShapeBlock.forward = _forward  # type: ignore


def _set_torch_determinism(seed: int) -> None:
    """Best-effort determinism for repeatable runs."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ShapeFormerAeonClassifier(BaseClassifier):
    """aeon-compatible classifier wrapper for ShapeFormer.

    Parameters
    repo_path : str
        Path to the ShapeFormer repo directory containing utils.py, Models/, Shapelet/, Training.py.
    window_size : int
        Window size passed to shapelet discovery.
    num_shapelet : int
        Number of non-overlapping shapelets per class returned by discovery.
    num_pip : float
        Proportion of PIPs used by candidate extraction.
    processes : int
        Number of processes used in shapelet discovery.
    epochs : int
        Training epochs (fixed, no internal validation).
    batch_size : int
        Training batch size.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay.
    dropout : float
        Dropout used by the model.
    normalise : bool
        If True, apply the repo's per-channel mean/std normalisation (fit on training data).
    nan_strategy : {'zero', 'noise'}
        How to replace NaNs in X (useful if you pre-pad unequal-length series with NaNs).
    random_state : int
        Random seed.
    use_gpu : bool
        If True and CUDA is available, train on GPU.
    gpu_id : int
        GPU index used when use_gpu is True.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:predict_proba": True,
        "X_inner_type": ["numpy3D"],
    }

    def __init__(
        self,
        repo_path: str,
        window_size: int = 50,
        num_shapelet: int = 3,
        num_pip: float = 0.2,
        processes: int = 1,
        epochs: int = 200,
        batch_size: int = 8,
        lr: float = 5e-2,
        weight_decay: float = 5e-4,
        dropout: float = 0.4,
        normalise: bool = False,
        nan_strategy: str = "zero",
        random_state: int = 1,
        use_gpu: bool = True,
        gpu_id: int = 0,
        # model dims (defaults taken from repo main.py)
        len_w: int = 64,
        local_embed_dim: int = 48,
        local_pos_dim: int = 48,
        shape_embed_dim: int = 128,
        pos_embed_dim: int = 128,
        dim_ff: int = 256,
        num_heads: int = 4,
        sge: int = 0,
        verbose: bool = False,
    ):
        self.repo_path = repo_path
        self.window_size = window_size
        self.num_shapelet = num_shapelet
        self.num_pip = num_pip
        self.processes = processes

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.normalise = normalise
        self.nan_strategy = nan_strategy
        self.random_state = random_state

        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        self.len_w = len_w
        self.local_embed_dim = local_embed_dim
        self.local_pos_dim = local_pos_dim
        self.shape_embed_dim = shape_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.sge = sge

        self.verbose = verbose

        super().__init__()

    def _fit(self, X, y):  # noqa: N802
        import torch
        from torch.utils.data import DataLoader

        repo_path = os.path.abspath(self.repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from utils import dataset_class
        from Models.shapeformer import model_factory
        from Models.optimizers import get_optimizer
        from Models.loss import get_loss_module
        from Shapelet.mul_shapelet_discovery import ShapeletDiscover
        from Training import SupervisedTrainer

        _patch_shapeblock_forward_for_cpu()
        _set_torch_determinism(int(self.random_state))

        y_arr = np.asarray(y)
        self.classes_, y_enc = np.unique(y_arr, return_inverse=True)
        y_enc = y_enc.astype(np.int64)

        X3 = _check_X3(X)
        rng = np.random.default_rng(int(self.random_state))
        X3 = _fill_nans(X3, self.nan_strategy, rng)

        self.n_channels_ = int(X3.shape[1])
        self.max_len_ = int(X3.shape[2])

        if self.normalise:
            self.mean_, self.std_ = _compute_mean_std(X3)
            X_train = _apply_mean_std(X3, self.mean_, self.std_)
        else:
            self.mean_, self.std_ = None, None
            X_train = X3

        if self.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{int(self.gpu_id)}")
        else:
            device = torch.device("cpu")
        self.device_ = str(device)

        shapelet_discovery = ShapeletDiscover(
            window_size=int(self.window_size),
            num_pip=float(self.num_pip),
            processes=int(self.processes),
            len_of_ts=int(self.max_len_),
            dim=int(self.n_channels_),
        )

        shapelet_discovery.extract_candidate(train_data=X_train)
        shapelet_discovery.discovery(train_data=X_train, train_labels=y_enc)
        shapelets_info = shapelet_discovery.get_shapelet_info(number_of_shapelet=int(self.num_shapelet))

        sw = torch.tensor(shapelets_info[:, 3])
        sw = torch.softmax(sw * 20, dim=0) * sw.shape[0]
        shapelets_info[:, 3] = sw.cpu().numpy()

        shapelets = []
        for si in shapelets_info:
            sc = X_train[int(si[0]), int(si[5]), int(si[1]) : int(si[2])]
            shapelets.append(sc)

        config = {
            "Net_Type": ["Shapeformer"],
            "window_size": int(self.window_size),
            "num_shapelet": int(self.num_shapelet),
            "num_pip": float(self.num_pip),
            "processes": int(self.processes),
            "sge": int(self.sge),
            "len_w": int(self.len_w),
            "local_embed_dim": int(self.local_embed_dim),
            "local_pos_dim": int(self.local_pos_dim),
            "shape_embed_dim": int(self.shape_embed_dim),
            "pos_embed_dim": int(self.pos_embed_dim),
            "dim_ff": int(self.dim_ff),
            "num_heads": int(self.num_heads),
            "dropout": float(self.dropout),
            "len_ts": int(self.max_len_),
            "ts_dim": int(self.n_channels_),
            "shapelets_info": shapelets_info,
            "shapelets": shapelets,
            "Data_shape": X_train.shape,
            "num_labels": int(len(self.classes_)),
        }

        model = model_factory(config).to(device)

        optim_class = get_optimizer("RAdam")
        optimizer = optim_class(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        loss_module = get_loss_module()

        train_dataset = dataset_class(X_train, y_enc.astype(np.int32))
        train_loader = DataLoader(train_dataset, batch_size=int(self.batch_size), shuffle=True, pin_memory=False)

        trainer = SupervisedTrainer(
            model,
            train_loader,
            device,
            loss_module,
            optimizer,
            l2_reg=0,
            print_interval=1000000,
            console=bool(self.verbose),
            print_conf_mat=False,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(self.epochs), eta_min=1e-3)

        for epoch in range(int(self.epochs)):
            trainer.train_epoch(epoch)
            scheduler.step()

        model.eval()

        self.model_ = model
        self.config_ = config
        return self

    def _predict(self, X):  # noqa: N802
        proba = self._predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def _predict_proba(self, X):  # noqa: N802
        import torch
        from torch.utils.data import DataLoader

        repo_path = os.path.abspath(self.repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from utils import dataset_class

        X3 = _check_X3(X)
        rng = np.random.default_rng(int(self.random_state))
        X3 = _fill_nans(X3, self.nan_strategy, rng)

        if self.normalise and getattr(self, "mean_", None) is not None and getattr(self, "std_", None) is not None:
            X3 = _apply_mean_std(X3, self.mean_, self.std_)

        y_dummy = np.zeros((X3.shape[0],), dtype=np.int32)
        ds = dataset_class(X3, y_dummy)
        dl = DataLoader(ds, batch_size=int(self.batch_size), shuffle=False, pin_memory=False)

        device = torch.device(self.device_) if isinstance(self.device_, str) else self.device_

        self.model_.eval()
        all_probs = []
        with torch.no_grad():
            for batch in dl:
                x, _, _ = batch
                logits = self.model_(x.to(device), 0)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)


__all__ = ["ShapeFormerAeonClassifier"]
