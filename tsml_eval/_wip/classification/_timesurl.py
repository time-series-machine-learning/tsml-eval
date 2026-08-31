"""Leakage-safe aeon adapter for the original TimesURL implementation."""

import random
from types import SimpleNamespace

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import check_random_state

from tsml_eval._wip.classification._timesurl_original.timesurl import TimesURL

__all__ = ["TimesURLClassifier"]


class TimesURLClassifier(BaseClassifier):
    """TimesURL self-supervised representation learner and classifier.

    This adapter follows the authors' UEA classification experiment: channels are
    standardised using training data only, a normalised time coordinate is appended,
    TimesURL is pretrained without labels, and an RBF SVM is fitted to full-series
    representations. The encoder, scaler, and downstream classifier are all fitted
    exclusively on the training collection.

    Parameters
    ----------
    output_dims : int, default=320
        Dimension of the learned representation.
    hidden_dims : int, default=64
        Hidden dimension of the encoder.
    depth : int, default=10
        Number of residual blocks in the encoder.
    n_iters : int or None, default=None
        Maximum training iterations. When both ``n_iters`` and ``n_epochs`` are
        ``None``, the original implementation selects 200 or 600 iterations based
        on the training-array size.
    n_epochs : int or None, default=None
        Maximum training epochs.
    batch_size : int, default=8
        Batch size. This is the default in the authors' experiment script; the
        underlying model class has a default of 16.
    learning_rate : float, default=1e-4
        Optimiser learning rate used by the authors' experiment script.
    max_train_length : int or None, default=3000
        Split longer training series into sections, as in the experiment script.
    temporal_unit : int, default=0
        Minimum temporal unit used by the contrastive loss.
    sgd : bool, default=False
        Use SGD and cosine scheduling instead of the original default AdamW.
    temperature : float, default=1.0
        Contrastive-loss temperature.
    lmd : float, default=0.01
        Weight of the hierarchical contrastive loss.
    segment_num : int, default=3
        Number of time intervals masked by the data collator.
    mask_ratio_per_seg : float, default=0.05
        Fraction of the series masked in each interval.
    eval_protocol : {"svm", "linear", "knn"}, default="svm"
        Downstream classifier. ``"svm"`` is used by the authors' classification
        command; the other protocols are also provided by their evaluation code.
    probe_n_jobs : int, default=5
        Parallel jobs used by the SVM grid search, matching the original code.
    probe_max_samples : int or None, default=None
        Optional representation subsample limit. ``None`` uses the original limits:
        10,000 for SVM and 100,000 for logistic regression.
    standardise : bool, default=True
        Standardise each channel with statistics fitted on the training set. This
        matches the original UEA loader.
    device : str, default="auto"
        PyTorch device. ``"auto"`` selects CUDA, then Apple MPS, then CPU.
    verbose : bool, default=False
        Print the original implementation's training progress.
    random_state : int or None, default=1234
        Seed used for Python, NumPy, and PyTorch randomness.

    Notes
    -----
    SVM decision scores are converted with a softmax for aeon's ``predict_proba``
    interface. They preserve the classifier decision but are not calibrated
    probabilities, matching the original use of ``SVC(probability=False)``.
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
        output_dims=320,
        hidden_dims=64,
        depth=10,
        n_iters=None,
        n_epochs=None,
        batch_size=8,
        learning_rate=1e-4,
        max_train_length=3000,
        temporal_unit=0,
        sgd=False,
        temperature=1.0,
        lmd=0.01,
        segment_num=3,
        mask_ratio_per_seg=0.05,
        eval_protocol="svm",
        probe_n_jobs=5,
        probe_max_samples=None,
        standardise=True,
        device="auto",
        verbose=False,
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
        self.sgd = sgd
        self.temperature = temperature
        self.lmd = lmd
        self.segment_num = segment_num
        self.mask_ratio_per_seg = mask_ratio_per_seg
        self.eval_protocol = eval_protocol
        self.probe_n_jobs = probe_n_jobs
        self.probe_max_samples = probe_max_samples
        self.standardise = standardise
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        """Fit the TimesURL encoder and downstream classifier."""
        import torch

        self._validate_hyperparameters()
        rng = check_random_state(self.random_state)
        self.random_state_ = int(rng.randint(np.iinfo(np.int32).max))
        random.seed(self.random_state_)
        np.random.seed(self.random_state_)
        torch.manual_seed(self.random_state_)

        self.device_ = self._resolve_device(torch)
        if str(self.device_).startswith("cuda"):
            torch.cuda.manual_seed_all(self.random_state_)

        self.n_channels_, self.n_timepoints_ = X.shape[1:]
        if self.n_timepoints_ < 4:
            raise ValueError("TimesURL requires series with at least 4 timepoints")

        train = np.transpose(np.asarray(X, dtype=np.float32), (0, 2, 1))
        self.scaler_ = None
        if self.standardise:
            self.scaler_ = StandardScaler().fit(
                train.reshape(-1, self.n_channels_)
            )

        train_data = self._make_timesurl_data(X)
        args = SimpleNamespace(
            lmd=self.lmd,
            segment_num=self.segment_num,
            mask_ratio_per_seg=self.mask_ratio_per_seg,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        self.encoder_ = TimesURL(
            input_dims=self.n_channels_,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            device=self.device_,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            sgd=self.sgd,
            max_train_length=self.max_train_length,
            temporal_unit=self.temporal_unit,
            args=args,
        )
        self.encoder_.fit(
            train_data,
            n_epochs=self.n_epochs,
            n_iters=self.n_iters,
            verbose=self.verbose,
            is_scheduler=self.sgd,
            temp=self.temperature,
        )

        representations = self._encode(X)
        labels = np.asarray([self._class_dictionary[value] for value in y])
        self.probe_ = self._fit_probe(representations, labels)
        return self

    def _validate_hyperparameters(self):
        if self.eval_protocol not in {"svm", "linear", "knn"}:
            raise ValueError(
                "eval_protocol must be one of {'svm', 'linear', 'knn'}, but found "
                f"{self.eval_protocol!r}"
            )
        if self.n_iters is not None and self.n_iters < 1:
            raise ValueError("n_iters must be a positive integer or None")
        if self.n_epochs is not None and self.n_epochs < 1:
            raise ValueError("n_epochs must be a positive integer or None")
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.max_train_length is not None and self.max_train_length < 4:
            raise ValueError("max_train_length must be at least 4 or None")

    def _resolve_device(self, torch):
        requested = "auto" if self.device is None else str(self.device)
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested.startswith("mps"):
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is None or not mps.is_available():
                raise RuntimeError("MPS was requested but is not available")
        return requested

    def _make_timesurl_data(self, X):
        values = np.transpose(np.asarray(X, dtype=np.float32), (0, 2, 1))
        if self.scaler_ is not None:
            shape = values.shape
            values = self.scaler_.transform(
                values.reshape(-1, self.n_channels_)
            ).reshape(shape)
        values = values.astype(np.float32, copy=False)
        n_cases, n_timepoints, _ = values.shape
        time = np.broadcast_to(
            np.linspace(0, 1, n_timepoints, dtype=np.float32)[None, :, None],
            (n_cases, n_timepoints, 1),
        )
        mask = np.ones(values.shape, dtype=np.float32)
        return {"x": np.concatenate([values, time], axis=2), "mask": mask}

    def _encode(self, X):
        data = self._make_timesurl_data(X)
        representations = self.encoder_.encode(
            data, encoding_window="full_series"
        )
        return representations.reshape(representations.shape[0], -1)

    def _fit_probe(self, features, y):
        if self.eval_protocol == "knn":
            return make_pipeline(
                StandardScaler(), KNeighborsClassifier(n_neighbors=1)
            ).fit(features, y)

        max_samples = self.probe_max_samples
        if max_samples is None:
            max_samples = 10_000 if self.eval_protocol == "svm" else 100_000
        if features.shape[0] > max_samples:
            features, _, y, _ = train_test_split(
                features,
                y,
                train_size=max_samples,
                random_state=0,
                stratify=y,
            )

        if self.eval_protocol == "linear":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    random_state=0,
                    max_iter=1_000_000,
                    multi_class="ovr",
                ),
            ).fit(features, y)

        svm = SVC(
            C=np.inf,
            gamma="scale",
        )
        n_classes = np.unique(y).shape[0]
        if features.shape[0] // n_classes < 5 or features.shape[0] < 50:
            return svm.fit(features, y)

        search = GridSearchCV(
            svm,
            {
                "C": [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000, np.inf],
                "kernel": ["rbf"],
                "degree": [3],
                "gamma": ["scale"],
                "coef0": [0],
                "shrinking": [True],
                "probability": [False],
                "tol": [0.001],
                "cache_size": [200],
                "class_weight": [None],
                "verbose": [False],
                "max_iter": [10_000_000],
                "decision_function_shape": ["ovr"],
                "random_state": [None],
            },
            cv=5,
            n_jobs=self.probe_n_jobs,
        )
        search.fit(features, y)
        return search.best_estimator_

    def _predict_proba(self, X):
        features = self._encode(X)
        if hasattr(self.probe_, "predict_proba"):
            return self.probe_.predict_proba(features)

        scores = self.probe_.decision_function(features)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def _predict(self, X):
        return self.classes_[self.probe_.predict(self._encode(X))]

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {
            "output_dims": 8,
            "hidden_dims": 8,
            "depth": 2,
            "n_iters": 2,
            "batch_size": 4,
            "eval_protocol": "linear",
            "device": "cpu",
            "random_state": 0,
        }
