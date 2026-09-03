"""XCM classifier for aeon.

Adapted from the authors' XCM implementation:
https://github.com/XAIseries/XCM

XCM is an explainable convolutional network for multivariate time series. Two
parallel branches see the input differently: a 2D branch convolves over time
within each channel separately, preserving which channel a feature came from,
and a 1D branch convolves over time across all channels at once. Their outputs
are concatenated, passed through a further 1D convolution, pooled and
classified. The separation is what makes the 2D branch's activations
attributable to individual channels, which is the basis of the paper's
explanations.

Unlike the other networks ported here this one is Keras rather than PyTorch,
following the authors, so it needs ``tensorflow`` rather than ``torch``.

The authors' training procedure is reproduced: a fixed number of epochs on the
whole training collection, with no validation split and no epoch selection.
Their ``main.py`` also runs a five fold cross validation, but only to report
per fold accuracies; the headline result comes from retraining on the full
training set, which is what this wrapper does.

One change was required. The original imports ``Conv1D`` and ``Conv2D`` from
``keras.layers.convolutional``, a path removed in Keras 3, so the imports come
from ``tensorflow.keras.layers`` instead. The layers and their arguments are
unchanged.

This wrapper is designed for aeon and therefore assumes input X is a 3D NumPy
array with shape (n_cases, n_channels, n_timepoints). The original expects
(n_cases, n_timepoints, n_channels, 1), so X is transposed and expanded
internally.

The original source is distributed under the MIT License.

MIT License

Copyright (c) 2021 Kevin Fauvel

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
__all__ = ["XCMClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.utils import check_random_state


#: The authors' grid, from §4.3: window size as a fraction of series length, and
#: batch size. They search the product of the two per dataset.
PAPER_WINDOW_SIZES = (0.2, 0.4, 0.6, 0.8, 1.0)
PAPER_BATCH_SIZES = (1, 8, 32)


def _as_grid(value):
    """Return a parameter as a list of candidates, scalar or sequence alike."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _build_xcm(n_timepoints, n_channels, n_classes, window_size, n_filters):
    """Build the authors' XCM network.

    A transcription of ``models/xcm.py``, with the layer imports taken from
    ``tensorflow.keras.layers`` because the original path was removed in
    Keras 3. Layer types, arguments, names and order are unchanged.
    """
    from tensorflow.keras.layers import (
        Activation,
        BatchNormalization,
        Conv1D,
        Conv2D,
        Dense,
        GlobalAveragePooling1D,
        Input,
        Reshape,
        concatenate,
    )
    from tensorflow.keras.models import Model

    n, k = n_timepoints, n_channels
    input_layer = Input(shape=(n, k, 1))

    # 2D branch: convolves along time within each channel, so an activation
    # stays attributable to the channel it came from
    a = Conv2D(
        filters=int(n_filters),
        kernel_size=(int(window_size * n), 1),
        strides=(1, 1),
        padding="same",
        name="2D",
    )(input_layer)
    a = BatchNormalization()(a)
    a = Activation("relu", name="2D_Activation")(a)
    a = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), name="2D_Reduced")(a)
    a = Activation("relu", name="2D_Reduced_Activation")(a)
    x = Reshape((n, k))(a)

    # 1D branch: convolves along time across all channels together
    b = Reshape((n, k))(input_layer)
    b = Conv1D(
        filters=int(n_filters),
        kernel_size=int(window_size * n),
        strides=1,
        padding="same",
        name="1D",
    )(b)
    b = BatchNormalization()(b)
    b = Activation("relu", name="1D_Activation")(b)
    b = Conv1D(filters=1, kernel_size=1, strides=1, name="1D_Reduced")(b)
    y = Activation("relu", name="1D_Reduced_Activation")(b)

    z = concatenate([x, y])
    z = Conv1D(
        filters=n_filters,
        kernel_size=int(window_size * n),
        strides=1,
        padding="same",
        name="1D_Final",
    )(z)
    z = BatchNormalization()(z)
    z = Activation("relu", name="1D_Final_Activation")(z)

    z = GlobalAveragePooling1D()(z)
    output_layer = Dense(n_classes, activation="softmax")(z)
    return Model(input_layer, output_layer)


class XCMClassifier(BaseClassifier):
    """Explainable Convolutional neural network for Multivariate time series.

    Two parallel branches see the input differently. The 2D branch convolves
    along time within each channel separately, so its activations remain
    attributable to individual channels. The 1D branch convolves along time
    across all channels together. The branches are concatenated, passed through
    a further 1D convolution, globally average pooled and classified.

    Parameters
    ----------
    window_size : float or sequence of float, default=0.8
        Length of the convolution kernels along time, as a fraction of the
        series length. Pass a single value to use it directly, or a sequence to
        select from it by cross-validation as the authors do. Their grid is
        :data:`PAPER_WINDOW_SIZES`, {0.2, 0.4, 0.6, 0.8, 1.0}; the default 0.8
        is the value their results table settles on most often, 13 of 30
        datasets. The 0.2 in their ``config.yml`` is the worked example for
        BasicMotions, not a default.
    max_window : int or None, default=100
        Upper bound on the kernel length in points, or None for no bound.
        Because ``window_size`` is a fraction, the kernel grows with the
        series: 0.8 of EigenWorms' 17984 points is a 14387 point kernel, and
        the authors do run kernels of that order, 40% of EigenWorms being 7193
        points. The bound is ours, not theirs, and keeps long series tractable;
        None reproduces their behaviour. The kernel is also floored at one
        point, since ``int(window_size * n)`` is zero for very short series.
    n_filters : int, default=128
        Number of filters in each convolution.
    n_epochs : int, default=100
        Training epochs. The authors train for a fixed number with no early
        stopping.
    batch_size : int or sequence of int, default=32
        Training batch size, tuned alongside ``window_size`` when a sequence is
        given. The authors' grid is :data:`PAPER_BATCH_SIZES`, {1, 8, 32}, but
        32 is their choice on 26 of 30 datasets and batch 1 costs roughly 32
        times the gradient steps, so tuning the window alone recovers most of
        the benefit for a small fraction of the compute.
    cv_folds : int, default=5
        Folds in the stratified cross-validation used to select parameters,
        following the authors' five. Reduced automatically when a class has
        fewer members than this, and selection is skipped entirely when the
        rarest class appears once.
    verbose : bool, default=False
        Whether Keras prints training progress.
    random_state : int, RandomState instance or None, default=None
        Seed controlling weight initialisation and batch shuffling.

    Attributes
    ----------
    model_ : keras.Model
        The fitted network.
    history_ : dict
        Keras training history, one entry per metric.
    window_size_ : int
        Kernel length along time actually used, in points.
    window_fraction_ : float
        Fraction selected, before conversion to points and any ``max_window``
        bound.
    batch_size_ : int
        Batch size actually used.
    cv_results_ : list of dict
        One entry per grid point, with its mean and per-fold accuracy. Empty
        when no search was run.
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
    .. [1] Fauvel, K., Lin, T., Masson, V., Fromont, E. and Termier, A. "XCM: An
       Explainable Convolutional Neural Network for Multivariate Time Series
       Classification." Mathematics, 9(23), 2021.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from multiverse.classification import XCMClassifier
    >>> X, y = make_example_3d_numpy(n_cases=8, n_channels=2, n_timepoints=20)
    >>> clf = XCMClassifier(n_epochs=2)  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP

    Selecting the window by cross-validation, as the paper does:

    >>> from multiverse.classification._xcm import PAPER_WINDOW_SIZES
    >>> clf = XCMClassifier(window_size=PAPER_WINDOW_SIZES)  # doctest: +SKIP
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        window_size=0.8,
        max_window: int | None = 100,
        n_filters: int = 128,
        n_epochs: int = 100,
        batch_size=32,
        cv_folds: int = 5,
        verbose: bool = False,
        random_state=None,
    ):
        self.window_size = window_size
        self.max_window = max_window
        self.n_filters = n_filters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _validate_parameters(self) -> None:
        """Check constructor parameters before any work is done."""
        for window in _as_grid(self.window_size):
            if not 0 < window <= 1:
                raise ValueError("window_size must be in (0, 1]")
        for batch in _as_grid(self.batch_size):
            if not isinstance(batch, int) or batch <= 0:
                raise ValueError("batch_size must be a positive integer")
        for name in ["n_filters", "n_epochs"]:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_window is not None and (
            not isinstance(self.max_window, int) or self.max_window <= 0
        ):
            raise ValueError("max_window must be a positive integer or None")
        if not isinstance(self.cv_folds, int) or self.cv_folds < 2:
            raise ValueError("cv_folds must be an integer of at least 2")

    @staticmethod
    def _to_original_layout(X: np.ndarray) -> np.ndarray:
        """Convert an aeon collection to the authors' (case, time, channel, 1)."""
        series = np.transpose(np.asarray(X, dtype=np.float32), (0, 2, 1))
        return series[..., np.newaxis]

    def _kernel_length(self, window_size: float) -> int:
        """Kernel length along time in points, for a fraction of the series.

        The authors compute ``int(window_size * n)``, which is zero for very
        short series and unbounded for long ones. The floor at one point is
        needed for the first; ``max_window`` bounds the second, and is ours,
        not theirs, so setting it to None reproduces their behaviour.
        """
        length = max(1, int(window_size * self.n_timepoints_))
        return length if self.max_window is None else min(length, self.max_window)

    def _fit_model(self, X, one_hot, window_size, batch_size, epochs):
        """Build, compile and train one network. Returns the fitted model."""
        # _build_xcm takes a fraction, as the authors' function does, so convert
        # back. The half point guards against int() rounding the fraction down.
        effective = (self._kernel_length(window_size) + 0.5) / self.n_timepoints_
        model = _build_xcm(
            self.n_timepoints_,
            self.n_channels_,
            self.n_classes_,
            effective,
            self.n_filters,
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        history = model.fit(
            self._to_original_layout(X),
            one_hot,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 if self.verbose else 0,
        )
        return model, history

    def _select_parameters(self, X, encoded_y, one_hot, grid, seed):
        """Choose window and batch size by the authors' cross-validation.

        Section 4.3: "hyperparameters were set by grid search based on the best
        average accuracy following a stratified 5-fold cross-validation on the
        training set". The selection therefore never sees the test data.

        Folds with fewer members than ``cv_folds`` cannot be stratified, so the
        number of splits is reduced to the smallest class count, and a class
        appearing once leaves nothing to select on, in which case the first
        candidate is taken.
        """
        from sklearn.model_selection import StratifiedKFold

        counts = np.bincount(encoded_y, minlength=self.n_classes_)
        splits = min(self.cv_folds, int(counts[counts > 0].min()))
        if splits < 2:
            self.cv_results_ = []
            return grid[0]

        folds = list(
            StratifiedKFold(
                n_splits=splits, shuffle=True, random_state=seed
            ).split(X, encoded_y)
        )

        self.cv_results_ = []
        for window_size, batch_size in grid:
            scores = []
            for train_index, test_index in folds:
                model, _ = self._fit_model(
                    X[train_index],
                    one_hot[train_index],
                    window_size,
                    batch_size,
                    self.n_epochs,
                )
                predicted = model.predict(
                    self._to_original_layout(X[test_index]), verbose=0
                ).argmax(axis=1)
                scores.append(float((predicted == encoded_y[test_index]).mean()))
                del model
            self.cv_results_.append(
                {
                    "window_size": window_size,
                    "batch_size": batch_size,
                    "mean_accuracy": float(np.mean(scores)),
                    "fold_accuracies": scores,
                }
            )
            if self.verbose:
                print(
                    f"window_size={window_size} batch_size={batch_size}: "
                    f"{np.mean(scores):.4f}"
                )

        best = max(self.cv_results_, key=lambda r: r["mean_accuracy"])
        return best["window_size"], best["batch_size"]

    def _fit(self, X: np.ndarray, y):
        self._validate_parameters()

        import tensorflow as tf

        rng = check_random_state(self.random_state)
        seed = int(rng.randint(np.iinfo(np.int32).max))
        tf.keras.utils.set_random_seed(seed)

        self.n_channels_, self.n_timepoints_ = X.shape[1], X.shape[2]

        encoded_y = np.asarray(
            [self._class_dictionary[label] for label in y], dtype=np.int64
        )
        one_hot = np.eye(self.n_classes_, dtype=np.float32)[encoded_y]

        grid = [
            (window, batch)
            for window in _as_grid(self.window_size)
            for batch in _as_grid(self.batch_size)
        ]
        if len(grid) > 1:
            window_size, batch_size = self._select_parameters(
                X, encoded_y, one_hot, grid, seed
            )
        else:
            window_size, batch_size = grid[0]
            self.cv_results_ = []

        self.window_size_ = self._kernel_length(window_size)
        self.batch_size_ = batch_size
        self.window_fraction_ = window_size

        # The authors refit on the whole training set once the parameters are
        # chosen, which is what produces their reported result.
        self.model_, history = self._fit_model(
            X, one_hot, window_size, batch_size, self.n_epochs
        )
        self.history_ = history.history
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
        return self.model_.predict(self._to_original_layout(X), verbose=0)

    def _predict(self, X: np.ndarray):
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict:
        """Return a small parameter set for aeon estimator checks."""
        return {
            "window_size": 0.8,
            "n_filters": 4,
            "n_epochs": 2,
            "batch_size": 4,
            "random_state": 0,
        }
