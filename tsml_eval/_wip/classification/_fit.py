"""FIT (Forest based on Interval Transformation) classifier.

A from-paper port of

    G. Li, S. Xu, S. Wang and P. S. Yu, "Forest based on Interval Transformation
    (FIT): A time series classifier with adaptive features", Expert Systems With
    Applications 213 (2023) 118923. doi:10.1016/j.eswa.2022.118923.

No reference implementation was released (the authors' code is Java and private),
so this follows Algorithms 1-5 and Sections 3.1-3.5 of the paper directly. The
correctness target is the per-dataset accuracy column of the paper's Table 1.

The method is an interval forest (TSF/CIF family) with an adaptive front-end:

1. Six representations of the series are formed (Section 3.1/3.2.1): the original
   series, first- and second-order difference, autocorrelation (ACF), the
   discrete Fourier transform coefficients (DFT), and the power spectrum (PS).
2. Six interval features are used (Section 3.1): mean, standard deviation, slope
   (least-squares regression slope), interquartile range, maximum and minimum.
3. AdaptiveFeatureSelection (Algorithm 1): 5x10-fold cross-validation with small
   ``k_CvTree``-tree forests scores every representation; representations whose
   CV accuracy is within ``ratio`` of the best are kept, and within those, a
   feature is kept if its decision-tree split count is at least the mean split
   count of the six features for that representation.
4. TrainingFIT (Algorithm 4): a ``n_estimators``-tree forest is trained using only
   the selected (representation, feature) pairs, each tree drawing sqrt(length)
   random intervals per pair with the whole series as a fixed first interval.

DFT/PS encoding note (underspecified in the paper): the DFT representation is the
real and imaginary parts of ``numpy.fft.rfft`` (the non-redundant half of the
Hermitian-symmetric transform, phase preserved); the power spectrum is the
periodogram ``|rfft|**2`` (phase discarded). These are complementary, which is
the only reading under which listing both representations adds information.
"""

__maintainer__ = []
__all__ = ["FITClassifier"]

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier

# Interval feature names, in the fixed order used for the split-count mapping.
_FEATURES = ("mean", "stdev", "slope", "iqr", "max", "min")
_N_FEATURES = len(_FEATURES)

# Representation names, in the fixed order used throughout.
_REPRESENTATIONS = ("original", "diff1", "diff2", "acf", "dft", "ps")


def _transform_representation(x2d, name):
    """Return the named representation of a 2D array ``(n_instances, length)``."""
    if name == "original":
        return x2d
    if name == "diff1":
        return np.diff(x2d, axis=1)
    if name == "diff2":
        return np.diff(x2d, n=2, axis=1)
    if name == "acf":
        return _acf(x2d)
    if name == "dft":
        coeffs = np.fft.rfft(x2d, axis=1)
        return np.concatenate([coeffs.real, coeffs.imag], axis=1)
    if name == "ps":
        coeffs = np.fft.rfft(x2d, axis=1)
        return (coeffs.real**2 + coeffs.imag**2)
    raise ValueError(f"unknown representation {name!r}")


def _acf(x2d, max_lag=None):
    """Normalised autocorrelation for lags 1..max_lag (default length-1)."""
    n, m = x2d.shape
    if max_lag is None:
        max_lag = m - 1
    max_lag = max(1, min(max_lag, m - 1))
    xc = x2d - x2d.mean(axis=1, keepdims=True)
    denom = (xc**2).sum(axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    out = np.empty((n, max_lag), dtype=np.float64)
    for k in range(1, max_lag + 1):
        out[:, k - 1] = (xc[:, : m - k] * xc[:, k:]).sum(axis=1)
    return out / denom


def _build_intervals(length, rng):
    """Random intervals for one tree (Algorithm 3): whole series then randoms.

    ``round(sqrt(length))`` intervals are returned. The first is the whole series
    ``[0, length)``. Each remaining interval has a random start in ``[0, length-3]``
    and a random length in ``[3, length//2]``, clamped to the series end.
    """
    n_intervals = max(1, int(round(np.sqrt(length))))
    intervals = [(0, length)]
    for _ in range(n_intervals - 1):
        if length < 6:  # no room for a proper random sub-interval
            intervals.append((0, length))
            continue
        start = rng.randint(0, length - 3 + 1)
        seg_len = rng.randint(3, length // 2 + 1)
        end = min(start + seg_len, length)
        intervals.append((start, end))
    return intervals


def _interval_features(rep, intervals, feature_ids):
    """Feature matrix for a representation over intervals.

    Columns are ordered interval-major, feature-minor, so column
    ``j * len(feature_ids) + f`` is feature ``feature_ids[f]`` on interval ``j``.
    """
    cols = []
    for start, end in intervals:
        seg = rep[:, start:end]
        width = end - start
        for fid in feature_ids:
            cols.append(_one_feature(seg, width, fid))
    return np.column_stack(cols)


def _one_feature(seg, width, fid):
    """One interval feature over slice ``seg`` of width ``width``."""
    name = _FEATURES[fid]
    if name == "mean":
        return seg.mean(axis=1)
    if name == "stdev":
        return seg.std(axis=1)
    if name == "max":
        return seg.max(axis=1)
    if name == "min":
        return seg.min(axis=1)
    if name == "iqr":
        q = np.percentile(seg, [25, 75], axis=1)
        return q[1] - q[0]
    if name == "slope":
        if width < 2:
            return np.zeros(seg.shape[0])
        x = np.arange(width, dtype=np.float64)
        xc = x - x.mean()
        denom = (xc**2).sum()
        return (seg @ xc) / denom
    raise ValueError(f"unknown feature id {fid}")


class FITClassifier(BaseClassifier):
    """Forest based on Interval Transformation (FIT).

    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees in the final forest (``k_Tree`` in the paper).
    cv_n_estimators : int, default=10
        Trees per fold in the cross-validation selection stage (``k_CvTree``).
    cv_repeats : int, default=5
        Number of times the 10-fold cross-validation is repeated.
    cv_folds : int, default=10
        Folds per cross-validation repeat.
    ratio : float, default=0.9
        A representation is kept if its CV accuracy is at least ``ratio`` times
        the best representation's CV accuracy.
    acf_max_lag : int or None, default=None
        Maximum ACF lag; ``None`` uses the full length-1 ACF series.
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    selected_pairs_ : list of (int, int)
        The kept ``(representation_index, feature_index)`` pairs.
    cv_accuracy_ : ndarray of shape (6,)
        Cross-validation accuracy of each representation.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:multithreading": False,
        "algorithm_type": "interval",
        "X_inner_type": "numpy3D",
    }

    def __init__(
        self,
        n_estimators=200,
        cv_n_estimators=10,
        cv_repeats=5,
        cv_folds=10,
        ratio=0.9,
        acf_max_lag=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.cv_n_estimators = cv_n_estimators
        self.cv_repeats = cv_repeats
        self.cv_folds = cv_folds
        self.ratio = ratio
        self.acf_max_lag = acf_max_lag
        self.random_state = random_state
        super().__init__()

    # -- fit ---------------------------------------------------------------

    def _fit(self, X, y):
        rng = check_random_state(self.random_state)
        x2d = X[:, 0, :]
        self._classes = np.unique(y)
        y_enc = np.searchsorted(self._classes, y)

        reps = self._all_representations(x2d)
        self.cv_accuracy_, split_counts = self._score_representations(reps, y_enc, rng)
        self.selected_pairs_ = self._select_pairs(self.cv_accuracy_, split_counts)

        # Fall back to every pair if selection is somehow empty (degenerate data).
        if not self.selected_pairs_:
            self.selected_pairs_ = [
                (i, f) for i in range(len(_REPRESENTATIONS)) for f in range(_N_FEATURES)
            ]

        self._forest = self._build_forest(reps, y_enc, self.selected_pairs_, rng)
        return self

    def _all_representations(self, x2d):
        return [_transform_representation(x2d, name) for name in _REPRESENTATIONS]

    def _score_representations(self, reps, y_enc, rng):
        """Algorithms 1-2: CV accuracy and split counts per representation."""
        n_reps = len(reps)
        cv_acc = np.zeros(n_reps)
        split_counts = np.zeros((n_reps, _N_FEATURES))
        all_feats = list(range(_N_FEATURES))

        # A stratified k-fold needs at least ``cv_folds`` members of the smallest
        # class; shrink the fold count for tiny datasets so selection still runs.
        min_class = np.min(np.bincount(y_enc))
        n_folds = max(2, min(self.cv_folds, min_class))

        for rep_idx, rep in enumerate(reps):
            length = rep.shape[1]
            fold_accs = []
            for repeat in range(self.cv_repeats):
                skf = StratifiedKFold(
                    n_splits=n_folds,
                    shuffle=True,
                    random_state=rng.randint(np.iinfo(np.int32).max),
                )
                for tr, val in skf.split(rep, y_enc):
                    acc = self._cv_fold(
                        rep[tr],
                        y_enc[tr],
                        rep[val],
                        y_enc[val],
                        length,
                        all_feats,
                        split_counts[rep_idx],
                        rng,
                    )
                    fold_accs.append(acc)
            cv_acc[rep_idx] = np.mean(fold_accs)
        return cv_acc, split_counts

    def _cv_fold(self, rep_tr, y_tr, rep_val, y_val, length, feats, counts, rng):
        """One CV fold: a small forest voted over, accumulating split counts."""
        votes = np.zeros((rep_val.shape[0], len(self._classes)), dtype=np.int64)
        for _ in range(self.cv_n_estimators):
            intervals = _build_intervals(length, rng)
            xt = _interval_features(rep_tr, intervals, feats)
            tree = DecisionTreeClassifier(
                criterion="entropy",
                max_features="sqrt",
                random_state=rng.randint(np.iinfo(np.int32).max),
            )
            tree.fit(xt, y_tr)
            # Split-count per feature: attribute index is interval-major,
            # feature-minor, so the feature is ``attr % _N_FEATURES``.
            used = tree.tree_.feature
            used = used[used >= 0]
            if used.size:
                np.add.at(counts, used % _N_FEATURES, 1)
            xv = _interval_features(rep_val, intervals, feats)
            preds = tree.classes_[tree.predict(xv)]
            votes[np.arange(preds.size), preds] += 1
        pred = votes.argmax(axis=1)
        return np.mean(pred == y_val)

    def _select_pairs(self, cv_acc, split_counts):
        """Algorithm 1 selection: representations by CV, features by split count."""
        threshold = self.ratio * cv_acc.max()
        pairs = []
        for rep_idx in range(len(cv_acc)):
            if cv_acc[rep_idx] < threshold:
                continue
            counts = split_counts[rep_idx]
            mean_count = counts.mean()
            for f in range(_N_FEATURES):
                if counts[f] >= mean_count:
                    pairs.append((rep_idx, f))
        return pairs

    def _build_forest(self, reps, y_enc, pairs, rng):
        """Algorithm 4: final forest over the selected (rep, feature) pairs."""
        forest = []
        for _ in range(self.n_estimators):
            plan = []  # (rep_idx, feature_idx, intervals)
            blocks = []
            for rep_idx, feat_idx in pairs:
                length = reps[rep_idx].shape[1]
                intervals = _build_intervals(length, rng)
                plan.append((rep_idx, feat_idx, intervals))
                blocks.append(
                    _interval_features(reps[rep_idx], intervals, [feat_idx])
                )
            xt = np.hstack(blocks)
            tree = DecisionTreeClassifier(
                criterion="entropy",
                max_features="sqrt",
                random_state=rng.randint(np.iinfo(np.int32).max),
            )
            tree.fit(xt, y_enc)
            forest.append((tree, plan))
        return forest

    # -- predict -----------------------------------------------------------

    def _predict(self, X):
        return self._classes[self._predict_proba(X).argmax(axis=1)]

    def _predict_proba(self, X):
        x2d = X[:, 0, :]
        reps = self._all_representations(x2d)
        votes = np.zeros((x2d.shape[0], len(self._classes)), dtype=np.int64)
        for tree, plan in self._forest:
            blocks = [
                _interval_features(reps[rep_idx], intervals, [feat_idx])
                for rep_idx, feat_idx, intervals in plan
            ]
            xt = np.hstack(blocks)
            preds = tree.classes_[tree.predict(xt)]
            votes[np.arange(preds.size), preds] += 1
        return votes / votes.sum(axis=1, keepdims=True)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {
            "n_estimators": 5,
            "cv_n_estimators": 2,
            "cv_repeats": 1,
            "cv_folds": 2,
        }
