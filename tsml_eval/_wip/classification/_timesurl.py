"""Leakage-safe aeon adapter for the TimesURL representation learner."""

import sys
from pathlib import Path

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state

_ORIGINAL = Path(__file__).with_name("_timesurl_original")
if str(_ORIGINAL) not in sys.path:
    sys.path.insert(0, str(_ORIGINAL))
from timesurl import TimesURL  # noqa: E402

__all__ = ["TimesURLClassifier"]


class TimesURLClassifier(BaseClassifier):
    """TimesURL self-supervised pretraining followed by a linear probe.

    The encoder is fitted exclusively on the training collection; test series are
    encoded only after pretraining and never participate in augmentation, masking,
    normalisation, or probe fitting.
    """

    _tags = {"X_inner_type": "numpy3D", "capability:multivariate": True,
             "algorithm_type": "deeplearning", "non_deterministic": True,
             "cant_pickle": True, "python_dependencies": "torch"}

    def __init__(self, output_dims=320, hidden_dims=64, depth=10,
                 n_iters=200, batch_size=16, learning_rate=1e-3,
                 device="auto", verbose=False, random_state=1234):
        self.output_dims = output_dims; self.hidden_dims = hidden_dims
        self.depth = depth; self.n_iters = n_iters; self.batch_size = batch_size
        self.learning_rate = learning_rate; self.device = device
        self.verbose = verbose; self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        rng = check_random_state(self.random_state)
        seed = int(rng.randint(np.iinfo(np.int32).max))
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        self.device_ = device
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        # Official TimesURL uses (case, time, channel), with an explicit time
        # coordinate and observation mask. These are derived from training data only.
        train = np.transpose(np.asarray(X, dtype=np.float32), (0, 2, 1))
        n, t, c = train.shape
        time = np.broadcast_to(np.linspace(0, 1, t, dtype=np.float32)[None, :, None], (n, t, 1))
        values = np.concatenate([train, time], axis=2)
        mask = np.ones((n, t, c), dtype=np.float32)
        from types import SimpleNamespace
        args = SimpleNamespace(lmd=0.01, segment_num=3, mask_ratio_per_seg=0.05,
                               batch_size=self.batch_size)
        self.encoder_ = TimesURL(c, self.output_dims, self.hidden_dims, self.depth,
                                 device=device, lr=self.learning_rate,
                                 batch_size=self.batch_size, args=args)
        self.encoder_.fit({"x": values, "mask": mask}, n_iters=self.n_iters,
                          verbose=self.verbose, is_scheduler=False)
        z = self._encode(X)
        z = z.reshape(z.shape[0], -1)
        yy = np.asarray([self._class_dictionary[v] for v in y])
        self.probe_ = LogisticRegression(max_iter=1000, random_state=seed,
                                         multi_class="auto").fit(z, yy)
        self.n_channels_, self.n_timepoints_ = X.shape[1:]
        return self

    def _encode(self, X):
        train = np.transpose(np.asarray(X, dtype=np.float32), (0, 2, 1))
        n, t, _ = train.shape
        time = np.broadcast_to(np.linspace(0, 1, t, dtype=np.float32)[None, :, None], (n, t, 1))
        values = np.concatenate([train, time], axis=2)
        mask = np.ones((n, t, values.shape[2] - 1), dtype=np.float32)
        z = self.encoder_.encode({"x": values, "mask": mask}, encoding_window="full_series")
        return z.reshape(z.shape[0], -1)

    def _predict_proba(self, X):
        return self.probe_.predict_proba(self._encode(X))

    def _predict(self, X):
        return self.classes_[self.probe_.predict(self._encode(X))]

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"output_dims": 8, "hidden_dims": 8, "depth": 2,
                "n_iters": 2, "batch_size": 4, "device": "cpu", "random_state": 0}
