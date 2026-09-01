"""Disjoint-CNN classifier for aeon.

Adapted from the authors' Disjoint-CNN implementation:
https://github.com/Navidfoumani/Disjoint-CNN

Disjoint-CNN factorises a multivariate convolution into two disjoint steps: a
temporal convolution applied within each channel, then a spatial convolution
across channels, with a non-linearity between them. Stacking these 1+1D blocks
is the paper's alternative to a single joint convolution over both axes.

This exists alongside ``aeon.classification.deep_learning.DisjointCNNClassifier``
deliberately. That implementation scores far below the authors' published
numbers, 20.3 accuracy points below on the 23 shared UEA datasets in our runs
(aeon issue #3775). This port follows the authors' own training procedure rather
than aeon's, so the two can be run against each other and the gap attributed.
The training differences are recorded under "Deviations from aeon" below; the
data pipeline is not one of them, since the authors pass ``normalise=False`` in
``Main.py`` and so train on raw series, as aeon does.

This wrapper is designed for aeon and therefore assumes input X is a 3D NumPy
array with shape (n_cases, n_channels, n_timepoints). The original expects
(n_cases, n_timepoints, n_channels, 1), so X is transposed and expanded
internally.

The original source is distributed under the MIT License.

MIT License

Copyright (c) 2021 Navid Mohammadi Foumani

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
__all__ = ["DisjointCNNClassifier"]

import math

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.utils import check_random_state

# Temporal kernel length per block, from the authors' DCNN_2L, DCNN_3L and
# DCNN_4L. The kernel shortens as the stack deepens.
KERNEL_SIZES = {2: (8, 5), 3: (8, 5, 3), 4: (8, 5, 5, 3)}
# Pool size after the last block, which also differs per variant.
POOL_SIZES = {2: 3, 3: 3, 4: 5}


def _build_disjoint_cnn(n_timepoints, n_channels, n_classes, n_layers, n_filters):
    """Build the authors' Disjoint-CNN network.

    A transcription of ``classifiers/DCNN_{2,3,4}L.py``. Each block is a
    temporal convolution over time within a channel, then a spatial convolution
    that spans every channel at once, then a permute that puts the filter axis
    back where the next block's spatial convolution expects it.

    The authors' arrays are (case, time, channel), from
    ``utils/data_loader.py::process_ts_data``, so time is axis 1 and the channel
    axis is the one the spatial convolution spans entirely.
    """
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Dense,
        ELU,
        GlobalAveragePooling2D,
        Input,
        MaxPooling2D,
        Permute,
    )
    from tensorflow.keras.models import Model

    kernels = KERNEL_SIZES[n_layers]
    input_layer = Input((n_timepoints, n_channels, 1))

    x = input_layer
    for block, kernel in enumerate(kernels):
        # Temporal: (kernel, 1) slides along time inside each channel.
        x = Conv2D(
            n_filters,
            (kernel, 1),
            strides=1,
            padding="same",
            kernel_initializer="he_uniform",
        )(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=1.0)(x)

        # Spatial: (1, width) spans the whole channel axis in one filter, so
        # padding is valid and the axis collapses to length 1.
        width = int(x.shape[2])
        x = Conv2D(
            n_filters,
            (1, width),
            strides=1,
            padding="valid",
            kernel_initializer="he_uniform",
        )(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=1.0)(x)

        # The authors permute after every block except the last.
        if block < len(kernels) - 1:
            x = Permute((1, 3, 2))(x)

    x = MaxPooling2D(pool_size=(POOL_SIZES[n_layers], 1), strides=None, padding="valid")(x)
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(n_classes, activation="softmax")(x)
    return Model(inputs=input_layer, outputs=output_layer)


def _class_weights(one_hot, mu=2.0):
    """Reproduce ``utils/classifier_tools.py::create_class_weight``.

    Weight for a class is ``log(mu * total / count)``, floored at 1.0, so rare
    classes are upweighted and common ones are never downweighted below 1.
    """
    counts = one_hot.sum(axis=0)
    total = counts.sum()
    weights = {}
    for index, count in enumerate(counts):
        score = math.log(mu * total / float(count)) if count > 0 else 1.0
        weights[index] = score if score > 1.0 else 1.0
    return weights


class DisjointCNNClassifier(BaseClassifier):
    """Disjoint-CNN, following the authors' training procedure.

    A stack of 1+1D blocks: a temporal convolution within each channel, then a
    spatial convolution across all channels, with ELU and batch normalisation
    between. The stack is max pooled, globally average pooled and classified.

    Parameters
    ----------
    n_layers : int, default=4
        Number of disjoint blocks, 2, 3 or 4, selecting the authors' DCNN_2L,
        DCNN_3L or DCNN_4L. The temporal kernel lengths and the final pool size
        follow the variant.
    n_filters : int, default=64
        Filters in every convolution.
    n_epochs : int, default=500
        Training epochs. The authors' ``Main.py`` sets 500.
    batch_size : int, default=8
        Upper bound on the batch size. The batch actually used is
        ``min(n_cases // 10, batch_size)``, the authors' rule, so small
        collections train with very small batches.
    validation_size : float, default=0.1
        Fraction of the training collection sampled to monitor validation loss.
        The authors sample this from the training set *with replacement* and do
        not hold it out, so it overlaps the training data. Reproduced because
        the learning rate schedule and the retained epoch both depend on it.
        Set to 0 to monitor training loss instead, which is what aeon does.
    use_class_weights : bool, default=True
        Whether to weight the loss by class, as the authors do. aeon does not.
    verbose : bool, default=False
        Whether Keras prints training progress.
    random_state : int, RandomState instance or None, default=None
        Seed controlling weight initialisation, batch shuffling and the
        validation sample.

    Attributes
    ----------
    model_ : keras.Model
        The fitted network, with the retained epoch's weights.
    history_ : dict
        Keras training history, one entry per metric.
    batch_size_ : int
        Batch size actually used, after the authors' rule.
    class_weights_ : dict or None
        Weights applied to the loss, or None when disabled.
    n_channels_ : int
        Number of channels seen in ``fit``.
    n_timepoints_ : int
        Series length seen in ``fit``.
    classes_ : np.ndarray
        Class labels, from ``BaseClassifier``.
    n_classes_ : int
        Number of classes, from ``BaseClassifier``.

    Notes
    -----
    Deviations from ``aeon.classification.deep_learning.DisjointCNNClassifier``,
    all of them places where aeon departs from the authors' code, and the
    candidates for the published-versus-obtained gap in aeon issue #3775:

    - **Class weighting.** The authors weight the loss per class; aeon does not.
    - **Batch size.** The authors use ``min(n_cases // 10, 8)``; aeon defaults
      to a flat 16, since ``use_mini_batch_size`` is False.
    - **Epochs.** The authors train for 500; aeon defaults to 2000.
    - **What is monitored.** The authors monitor validation loss for both the
      learning rate schedule and the retained epoch; aeon monitors training
      loss, having no validation split.

    The architecture here is the clean stack, matching aeon, so that a
    difference between the two is attributable to the training procedure alone.
    It is worth recording that the authors' own ``DCNN_4L.py`` does not build
    that stack: in the third block the spatial convolution is applied to
    ``conv2``, the second block's output, rather than to the third block's
    temporal convolution, which is therefore computed and discarded. It reads
    like a copy-paste slip, but it is the graph their published numbers came
    from, so replicating those exactly would need it. ``DCNN_2L`` and
    ``DCNN_3L`` are unaffected.

    References
    ----------
    .. [1] Foumani, S. N. M., Tan, C. W. and Salehi, M. "Disjoint-CNN for
       Multivariate Time Series Classification." ICDMW, 2021.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from multiverse.classification import DisjointCNNClassifier
    >>> X, y = make_example_3d_numpy(n_cases=8, n_channels=2, n_timepoints=20)
    >>> clf = DisjointCNNClassifier(n_epochs=2)  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
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
        n_layers: int = 4,
        n_filters: int = 64,
        n_epochs: int = 500,
        batch_size: int = 8,
        validation_size: float = 0.1,
        use_class_weights: bool = True,
        verbose: bool = False,
        random_state=None,
    ):
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.use_class_weights = use_class_weights
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _validate_parameters(self) -> None:
        """Check constructor parameters before any work is done."""
        if self.n_layers not in KERNEL_SIZES:
            raise ValueError(
                f"n_layers must be one of {sorted(KERNEL_SIZES)}, got {self.n_layers}"
            )
        for name in ["n_filters", "n_epochs", "batch_size"]:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.validation_size < 1:
            raise ValueError("validation_size must be in [0, 1)")

    @staticmethod
    def _to_original_layout(X: np.ndarray) -> np.ndarray:
        """Convert an aeon collection to the authors' (case, time, channel, 1)."""
        series = np.transpose(np.asarray(X, dtype=np.float32), (0, 2, 1))
        return series[..., np.newaxis]

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

        # The authors' rule, which can floor at zero on tiny collections.
        self.batch_size_ = max(1, min(X.shape[0] // 10, self.batch_size))

        self.model_ = _build_disjoint_cnn(
            self.n_timepoints_,
            self.n_channels_,
            self.n_classes_,
            self.n_layers,
            self.n_filters,
        )
        self.model_.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        data = self._to_original_layout(X)
        validation_data = None
        monitor = "loss"
        if self.validation_size:
            # Sampled from the training set with replacement and left in it,
            # exactly as the authors do.
            size = max(1, int(X.shape[0] * self.validation_size))
            index = rng.randint(0, X.shape[0], size)
            validation_data = (data[index], one_hot[index])
            monitor = "val_loss"

        self.class_weights_ = (
            _class_weights(one_hot) if self.use_class_weights else None
        )

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor, factor=0.5, patience=50, min_lr=0.0001
            ),
            # The authors checkpoint to disk and reload; restoring the best
            # weights in memory is the same selection without the file.
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=self.n_epochs,
                restore_best_weights=True,
            ),
        ]

        history = self.model_.fit(
            data,
            one_hot,
            validation_data=validation_data,
            class_weight=self.class_weights_,
            epochs=self.n_epochs,
            batch_size=self.batch_size_,
            verbose=1 if self.verbose else 0,
            callbacks=callbacks,
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
            "n_layers": 2,
            "n_filters": 4,
            "n_epochs": 2,
            "batch_size": 4,
            "validation_size": 0.0,
            "random_state": 0,
        }
