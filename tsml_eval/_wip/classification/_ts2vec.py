"""TS2Vec classifier for aeon.

Adapted from the authors' TS2Vec implementation:
https://github.com/zhihanyue/ts2vec

TS2Vec is a self-supervised representation learner, not an end-to-end
classifier. A hierarchical contrastive objective pretrains an encoder, the
collection is encoded to one vector per series, and a classifier is fitted on
those representations.

The authors' package is vendored under ``_ts2vec_original`` and imported as a
subpackage. The only change is that the sibling imports in ``ts2vec.py`` are
rewritten as relative imports; the architecture and training procedure are
untouched.

For the UEA archive the authors evaluate with a support vector machine, chosen
by grid search over C on the training representations (``train.py`` passes
``eval_protocol='svm'``), so that is the default here. Their grid sets
``probability=False``, which leaves an ``SVC`` unable to produce probability
estimates, so this wrapper sets it True: aeon classifiers must implement
``predict_proba``. That adds Platt scaling, fitted by internal cross-validation
on the training data only.

This wrapper is designed for aeon and therefore assumes input X is a 3D NumPy
array with shape (n_cases, n_channels, n_timepoints). The original expects
(n_cases, n_timepoints, n_channels), so X is transposed internally.

The original source is distributed under the MIT License.

MIT License

Copyright (c) 2022 Zhihan Yue

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
__all__ = ["TS2VecClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import check_random_state


class TS2VecClassifier(BaseClassifier):
    """TS2Vec self-supervised pretraining followed by a classifier probe.

    The encoder is pretrained on the training collection with the authors'
    hierarchical contrastive objective, the collection is encoded to one vector
    per series, and a probe is fitted on the representations. At prediction time
    the fitted encoder embeds the new series and the probe classifies them.

    The encoder is fitted exclusively on the training collection. Test series
    are encoded only after pretraining and take no part in the contrastive
    objective or in fitting the probe.

    Parameters
    ----------
    output_dims : int, default=320
        Width of the representation the encoder produces.
    hidden_dims : int, default=64
        Width of the encoder's hidden layers.
    depth : int, default=10
        Number of dilated convolution blocks in the encoder.
    n_iters : int or None, default=None
        Number of pretraining iterations. When both this and ``n_epochs`` are
        None the authors' default applies: 200 iterations for collections of
        100 series or fewer, 600 otherwise.
    n_epochs : int or None, default=None
        Number of pretraining epochs, as an alternative to ``n_iters``.
    batch_size : int, default=16
        Pretraining batch size.
    learning_rate : float, default=0.001
        Encoder learning rate.
    max_train_length : int, default=3000
        Series longer than this are split into sections during pretraining, as
        in the original.
    temporal_unit : int, default=0
        Minimum unit for temporal contrast, as in the original.
    probe_max_samples : int or None, default=None
        Cap on the number of cases the probe is fitted on. None takes the
        authors' values, 10000 for the SVM probe and 100000 for the logistic
        one, and the collection is subsampled with stratification when it is
        larger. Without this the SVM grid search is unaffordable on the large
        collections.
    probe : {"svm", "logistic"}, default="svm"
        Classifier fitted on the representations. ``"svm"`` reproduces the
        authors' UEA protocol, an ``SVC`` chosen by grid search over C on the
        training representations. ``"logistic"`` is their linear alternative,
        and is what the TimesURL wrapper uses, so it allows the two encoders to
        be compared without the probe differing.
    device : {"auto", "cpu", "cuda"} or torch device string, default="auto"
        Device used for pretraining and encoding.
    verbose : bool, default=False
        Whether the encoder prints pretraining progress.
    random_state : int, RandomState instance or None, default=1234
        Seed controlling encoder initialisation, cropping and the probe.

    Attributes
    ----------
    encoder_ : object
        Pretrained TS2Vec encoder.
    probe_ : object
        Classifier fitted on the encoded training collection.
    probe_cases_ : int
        Number of cases the probe was fitted on, after any subsampling.
    device_ : str
        Resolved device.
    n_channels_ : int
        Number of channels seen in ``fit``.
    n_timepoints_ : int
        Series length seen in ``fit``.
    classes_ : np.ndarray
        Class labels, from ``BaseClassifier``.
    n_classes_ : int
        Number of classes, from ``BaseClassifier``.

    References
    ----------
    .. [1] Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y. and Xu, B.
       "TS2Vec: Towards Universal Representation of Time Series." AAAI, 2022.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from multiverse.classification import TS2VecClassifier
    >>> X, y = make_example_3d_numpy(n_cases=8, n_channels=2, n_timepoints=20)
    >>> clf = TS2VecClassifier(n_iters=2, device="cpu")  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
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
        probe: str = "svm",
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
        self.probe = probe
        self.probe_max_samples = probe_max_samples
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _validate_parameters(self) -> None:
        """Check constructor parameters before any work is done."""
        for name in ["output_dims", "hidden_dims", "depth", "batch_size",
                     "max_train_length"]:
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
        if self.probe not in ("svm", "logistic"):
            raise ValueError(f'probe must be "svm" or "logistic", got {self.probe!r}')

    def _resolve_device(self) -> str:
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return self.device

    @staticmethod
    def _to_original_layout(X: np.ndarray) -> np.ndarray:
        """Convert an aeon collection to the authors' (case, time, channel)."""
        return np.transpose(np.asarray(X, dtype=np.float32), (0, 2, 1))

    def _subsample(self, features, y, seed):
        """Cap the collection the probe is fitted on, as the authors do.

        ``fit_svm`` takes ``MAX_SAMPLES=10000`` and ``fit_lr`` 100000, and both
        subsample with stratification before fitting. This is not a detail:
        the SVM grid is ten values of C over five folds, and an RBF SVC is
        between quadratic and cubic in the number of cases, so without the cap
        the probe alone can outrun a 60 hour job on the larger collections.
        """
        limit = self.probe_max_samples
        if limit is None:
            limit = 10_000 if self.probe == "svm" else 100_000
        if features.shape[0] <= limit:
            return features, y
        features, _, y, _ = train_test_split(
            features, y, train_size=limit, random_state=0, stratify=y
        )
        return features, y

    def _build_probe(self, n_cases: int, seed: int):
        """Return the probe, following the authors' evaluation protocols.

        The SVM path mirrors ``tasks/_eval_protocols.py::fit_svm``: a plain SVC
        on very small or very unbalanced collections, otherwise a grid search
        over C. ``probability=True`` is set so that ``predict_proba`` exists.
        """
        if self.probe == "logistic":
            # One-vs-rest, matching the authors' fit_lr and the TimesURL probe.
            # They pass multi_class="ovr", removed in scikit-learn 1.8; without
            # the wrapper this would quietly become multinomial and stop being
            # the same probe TimesURL uses.
            return make_pipeline(
                StandardScaler(),
                OneVsRestClassifier(
                    LogisticRegression(max_iter=1000000, random_state=seed)
                ),
            )

        svm = SVC(C=np.inf, gamma="scale", probability=True, random_state=seed)
        if n_cases // self.n_classes_ < 5 or n_cases < 50:
            return svm
        return GridSearchCV(
            svm,
            {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf],
             "kernel": ["rbf"], "gamma": ["scale"]},
            cv=5,
            n_jobs=1,
        )

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """Embed a collection with the fitted encoder, one vector per case."""
        return self.encoder_.encode(
            self._to_original_layout(X), encoding_window="full_series"
        )

    def _fit(self, X: np.ndarray, y):
        self._validate_parameters()

        import torch

        from multiverse.classification._ts2vec_original.ts2vec import TS2Vec

        rng = check_random_state(self.random_state)
        seed = int(rng.randint(np.iinfo(np.int32).max))
        np.random.seed(seed)
        torch.manual_seed(seed)

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
            # the authors' default, from train.py
            n_iters = 200 if X.shape[0] <= 100 else 600
        self.encoder_.fit(
            self._to_original_layout(X),
            n_epochs=self.n_epochs,
            n_iters=n_iters,
            verbose=self.verbose,
        )

        encoded = self._encode(X)
        encoded_y = np.asarray(
            [self._class_dictionary[label] for label in y], dtype=np.int64
        )
        fit_features, fit_y = self._subsample(encoded, encoded_y, seed)
        self.probe_cases_ = int(fit_features.shape[0])
        self.probe_ = self._build_probe(
            self.probe_cases_, seed
        ).fit(fit_features, fit_y)
        return self

    def _check_shape(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"X has {X.shape[1]} channels, but the classifier was fitted "
                f"with {self.n_channels_}."
            )
        if X.shape[2] != self.n_timepoints_:
            raise ValueError(
                f"X has length {X.shape[2]}, but the classifier was fitted with "
                f"length {self.n_timepoints_}."
            )

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_shape(X)
        return self.probe_.predict_proba(self._encode(X))

    def _predict(self, X: np.ndarray):
        self._check_shape(X)
        return self.classes_[self.probe_.predict(self._encode(X))]

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict:
        """Return a small parameter set for aeon estimator checks."""
        return {
            "output_dims": 8,
            "hidden_dims": 8,
            "depth": 2,
            "n_iters": 2,
            "batch_size": 4,
            "probe": "logistic",
            "device": "cpu",
            "random_state": 0,
        }
