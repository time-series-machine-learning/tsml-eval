"""RankSCL classifier for aeon.

Adapted from the authors' RankSCL implementation:
https://github.com/UConn-DSIS/Rank-Supervised-Contrastive-Learning-for-Time-Series-Classification

RankSCL is supervised contrastive learning with two changes to the usual
recipe. Positives are augmented in the embedding space rather than the input
space, by jittering the representation of a same-class neighbour, and the loss
is rank-based: for each positive it counts, softly, how many negatives sit
closer to the anchor than that positive does. Training produces an encoder;
classification is an SVM fitted on the encoder's representations, as in
TS2Vec, whose evaluation protocol this inherits.

The pieces are transcribed rather than vendored. The authors' modules import
each other absolutely and pull in matplotlib and a logging setup that writes to
a hard-coded path, none of which belongs in a library, and the parts that
matter are small: the FCN encoder, the embedding-space augmentation and the
ranking loss.

This wrapper is designed for aeon and therefore assumes input X is a 3D NumPy
array with shape (n_cases, n_channels, n_timepoints), which is the layout the
authors' encoder expects after their transpose, so no reordering is needed.

The original source is distributed under the MIT License.

MIT License

Copyright (c) 2024 UConn-DSIS

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
__all__ = ["RankSCLClassifier"]

import numpy as np
import torch
from aeon.classification import BaseClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

#: The authors' UEA settings, from ``scripts/uea.sh``.
PAPER_DEFAULTS = {
    "n_epochs": 100,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "weight_decay": 5e-4,
    "aug_positives": 5,
    "distance": "EU",
}


class _FCNEncoder(nn.Module):
    """The authors' FCN encoder.

    A transcription of ``models/FCN.py::FCN``. Three dilated convolution blocks
    widening 24 -> 64 -> 320, each with batch normalisation and ReLU, then
    global average pooling to a 320 dimensional representation, and a
    projection head of the same width used only during training.

    The dilations grow 2, 4, 8 while the kernels shrink 7, 5, 3, so the
    receptive field widens without the parameter count following it.

    Defined at module level rather than inside a factory so that fitted
    classifiers pickle, which aeon's estimator checks require.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, 64, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 320, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(320, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Linear(320, 320),
        )

    def forward(self, x):
        """Return the projected embedding and the representation."""
        representation = self.encoder(x)
        return self.projection(representation), representation


def _build_encoder(n_channels: int):
    """Return a new encoder for a collection with this many channels."""
    return _FCNEncoder(n_channels)


def _same_class_neighbour(embeddings, labels, generator):
    """Replace each embedding with that of a random same-class neighbour.

    ``utils/utils.py::generate_pos``. A case with no other member of its class
    in the batch keeps its own embedding, which makes it its own positive.
    """
    out = torch.zeros_like(embeddings)
    for position in range(embeddings.shape[0]):
        same = (labels == labels[position]).nonzero().flatten()
        same = same[same != position]
        if len(same) == 0:
            out[position] = embeddings[position]
        else:
            pick = torch.randint(
                len(same), (1,), generator=generator, device=same.device
            )
            out[position] = embeddings[same[pick]]
    return out


def _augment(embeddings, labels, n_positives, sigmas=(0.03, 0.05)):
    """Augment positives in the embedding space by jittering.

    ``utils/augmentation.py::aug_data``. The normalised embeddings are kept,
    and for each sigma ``n_positives`` jittered copies are appended, giving
    ``2 * n_positives + 1`` blocks in all with the labels repeated to match.

    Note that the jitter is applied to the unnormalised embeddings while the
    first block is normalised, which is what the authors do.
    """
    stacked = [functional.normalize(embeddings, dim=1)]
    for sigma in sigmas:
        for _ in range(n_positives):
            noise = torch.normal(
                mean=0.0, std=sigma, size=embeddings.shape, device=embeddings.device
            )
            stacked.append(embeddings + noise)
    repeats = 1 + len(sigmas) * n_positives
    return torch.cat(stacked, dim=0), labels.repeat(repeats)


def _ranking_loss(embeddings, labels, distance):
    """The paper's rank-based contrastive loss.

    ``loss/Ranking_loss.py::Ranking_loss``. For every anchor and every positive
    of that anchor, take the negatives that are at least as close to the anchor
    as the positive is, which are the ones ranked wrongly, and sum a sigmoid of
    how much closer they are. The per-positive sums are compressed with arctan
    and averaged, so a badly ranked positive saturates rather than dominating.

    Returns None when the batch has no anchor with both a positive and a
    negative, which the authors' version would raise on.
    """
    if distance == "Cosine":
        matrix = -torch.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
    else:
        matrix = torch.cdist(embeddings, embeddings, p=2)

    same = labels.reshape(1, -1) == labels.reshape(-1, 1)
    violations = []
    for anchor in range(matrix.shape[0]):
        negatives = matrix[anchor][~same[anchor]]
        if negatives.numel() == 0:
            continue
        positives = same[anchor].nonzero().flatten()
        for positive in positives[positives != anchor]:
            gap = matrix[anchor, positive]
            closer = negatives[negatives <= gap]
            violations.append(torch.sigmoid(gap - closer).sum())

    if not violations:
        return None
    return torch.atan(torch.stack(violations)).mean()


class RankSCLClassifier(BaseClassifier):
    """Rank Supervised Contrastive Learning for time series classification.

    An FCN encoder is trained with a supervised contrastive objective in which
    positives are augmented in the embedding space and the loss is rank-based,
    counting how many negatives intrude on each positive. The representations
    are then classified by an SVM, following the TS2Vec evaluation protocol the
    authors adopt.

    Parameters
    ----------
    n_epochs : int, default=100
        Encoder training epochs, the authors' ``epochs_up``.
    batch_size : int, default=4
        Training batch size. The authors use 4 for the UEA archive. Batches
        smaller than this are dropped, as in the original, so a collection with
        fewer cases than ``batch_size`` cannot be trained on.
    learning_rate : float, default=1e-4
        Adam learning rate.
    weight_decay : float, default=5e-4
        Adam weight decay.
    aug_positives : int, default=5
        Jittered copies generated per sigma, so the loss sees
        ``2 * aug_positives + 1`` blocks per batch.
    distance : {"EU", "Cosine"}, default="EU"
        Distance the ranking is computed over. The authors use Euclidean for
        the UEA archive.
    probe : {"svm", "logistic"}, default="svm"
        Classifier fitted on the representations, matching the protocol in
        ``utils/_eval_protocols.py``.
    probe_max_samples : int or None, default=None
        Cap on the cases the probe is fitted on. None takes the authors'
        values, 10000 for the SVM probe and 100000 for the logistic one, with
        stratified subsampling above that. Their ``fit_svm`` carries the same
        cap, inherited from TS2Vec.
    device : {"auto", "cpu", "cuda"} or torch device string, default="auto"
        Device used for training and encoding.
    verbose : bool, default=False
        Whether to print the loss every ten epochs, as the authors do.
    random_state : int, RandomState instance or None, default=None
        Seed controlling initialisation, batching, the neighbour draw and the
        jitter.

    Attributes
    ----------
    encoder_ : torch.nn.Module
        The trained encoder.
    probe_ : object
        Classifier fitted on the encoded training collection.
    probe_cases_ : int
        Number of cases the probe was fitted on, after any subsampling.
    history_ : list of dict
        Mean loss per epoch.
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
    .. [1] Ren, Q., Luo, D. and Song, D. "Rank Supervised Contrastive Learning
       for Time Series Classification." ICDM, 2024.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from multiverse.classification import RankSCLClassifier
    >>> X, y = make_example_3d_numpy(n_cases=8, n_channels=2, n_timepoints=20)
    >>> clf = RankSCLClassifier(n_epochs=2)  # doctest: +SKIP
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
        n_epochs: int = 100,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        aug_positives: int = 5,
        distance: str = "EU",
        probe: str = "svm",
        probe_max_samples: int | None = None,
        device: str = "auto",
        verbose: bool = False,
        random_state=None,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.aug_positives = aug_positives
        self.distance = distance
        self.probe = probe
        self.probe_max_samples = probe_max_samples
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _validate_parameters(self) -> None:
        """Check constructor parameters before any work is done."""
        for name in ["n_epochs", "batch_size"]:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.aug_positives, int) or self.aug_positives < 0:
            raise ValueError("aug_positives must be a non-negative integer")
        if self.learning_rate < 0 or self.weight_decay < 0:
            raise ValueError("learning_rate and weight_decay must be non-negative")
        if self.distance not in ("EU", "Cosine"):
            raise ValueError(f'distance must be "EU" or "Cosine", got {self.distance!r}')
        if self.probe not in ("svm", "logistic"):
            raise ValueError(f'probe must be "svm" or "logistic", got {self.probe!r}')

    def _resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return self.device

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """Return normalised encoder representations, as the probe expects."""
        self.encoder_.eval()
        with torch.no_grad():
            batch = torch.as_tensor(X, dtype=torch.float32, device=self.device_)
            _, representation = self.encoder_(batch)
            representation = functional.normalize(representation, dim=1)
        return representation.cpu().numpy()

    def _subsample(self, features, y):
        """Cap the collection the probe is fitted on, as the authors do."""
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
        """Return the probe, following ``utils/_eval_protocols.py``."""
        if self.probe == "logistic":
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

    def _fit(self, X: np.ndarray, y):
        self._validate_parameters()

        rng = check_random_state(self.random_state)
        seed = int(rng.randint(np.iinfo(np.int32).max))
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device_ = self._resolve_device()
        self.n_channels_, self.n_timepoints_ = X.shape[1], X.shape[2]

        encoded_y = np.asarray(
            [self._class_dictionary[label] for label in y], dtype=np.int64
        )
        if X.shape[0] < self.batch_size:
            raise ValueError(
                f"batch_size={self.batch_size} exceeds the {X.shape[0]} training "
                "cases, and the authors drop the last incomplete batch, so no "
                "batch would be formed. Reduce batch_size."
            )

        self.encoder_ = _build_encoder(self.n_channels_).to(self.device_)
        optimizer = torch.optim.Adam(
            self.encoder_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        generator = torch.Generator(device=self.device_).manual_seed(seed)
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(X, dtype=torch.float32),
                torch.as_tensor(encoded_y, dtype=torch.long),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.history_ = []
        for epoch in range(self.n_epochs):
            self.encoder_.train()
            losses = []
            for batch, labels in loader:
                batch = batch.to(self.device_)
                labels = labels.to(self.device_)
                optimizer.zero_grad()

                projected, _ = self.encoder_(batch)
                neighbours = _same_class_neighbour(projected, labels, generator)
                augmented, repeated = _augment(
                    neighbours, labels, self.aug_positives
                )
                # The anchors themselves replace the first block, so the loss
                # sees each anchor once and its jittered positives after it.
                augmented = torch.cat(
                    [functional.normalize(projected, dim=1),
                     augmented[projected.shape[0]:]],
                    dim=0,
                )
                loss = _ranking_loss(augmented, repeated, self.distance)
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            mean_loss = float(np.mean(losses)) if losses else float("nan")
            self.history_.append({"epoch": epoch + 1, "loss": mean_loss})
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch + 1} ----- loss {mean_loss:.3f}")

        representations = self._encode(X)
        fit_features, fit_y = self._subsample(representations, encoded_y)
        self.probe_cases_ = int(fit_features.shape[0])
        self.probe_ = self._build_probe(self.probe_cases_, seed).fit(
            fit_features, fit_y
        )
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
            "n_epochs": 2,
            "batch_size": 2,
            "aug_positives": 1,
            "probe": "logistic",
            "device": "cpu",
            "random_state": 0,
        }
