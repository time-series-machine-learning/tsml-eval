"""Shared interval feature transform for SharedDrCIF.

Computes the DrCIF attribute pool once over a fixed set of intervals on the
three DrCIF representations (base series, first order differences,
periodogram), producing a single feature matrix shared by all trees —
rather than DrCIF's per-tree random interval transforms.

Feature layout (documented for tests): for each representation, for each
interval: the 22 catch22 features per channel (channels concatenated), then
for each channel the summary statistics in _STAT_FUNCS order (plus the
quantile features for the "union35" pool).
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["SharedIntervalTransform", "dyadic_intervals", "drcif_random_intervals"]

import numpy as np
from sklearn.utils import check_random_state

from aeon.transformations.collection import PeriodogramTransformer
from aeon.transformations.collection.feature_based import Catch22
from aeon.utils.numba.general import first_order_differences_3d
from aeon.utils.numba.stats import (
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_slope,
    row_std,
)

from tsml_eval._wip.classification._quantile_stats import (
    row_quantile_10,
    row_quantile_25,
    row_quantile_25_centred,
    row_quantile_75,
    row_quantile_75_centred,
    row_quantile_90,
)

_STAT_FUNCS_29 = [
    row_mean,
    row_std,
    row_slope,
    row_median,
    row_iqr,
    row_numba_min,
    row_numba_max,
]
_STAT_FUNCS_35 = _STAT_FUNCS_29 + [
    row_quantile_10,
    row_quantile_25,
    row_quantile_75,
    row_quantile_90,
    row_quantile_25_centred,
    row_quantile_75_centred,
]


def dyadic_intervals(m, min_interval_length=3, max_depth=6):
    """QUANT-style dyadic interval grid with half-length shifts.

    Depth 0 is the full series; each depth halves the interval length and
    tiles the series, adding a half-shifted tiling for depths > 0. Stops when
    intervals would be shorter than min_interval_length or deeper than
    max_depth. Returns a list of (start, end) tuples, duplicates removed.
    """
    intervals = []
    seen = set()
    for d in range(max_depth + 1):
        w = m // (2**d)
        if w < min_interval_length:
            break
        starts = list(range(0, m - w + 1, w))
        if d > 0:
            h = w // 2
            starts += list(range(h, m - w + 1, w))
        for s in starts:
            if (s, s + w) not in seen:
                seen.add((s, s + w))
                intervals.append((s, s + w))
    return intervals


def drcif_random_intervals(m, n, rng, min_interval_length=3, max_interval_prop=0.5):
    """Random intervals matching aeon RandomIntervals / DrCIF's generation rule.

    For each interval, with equal probability either the start or the end is
    drawn uniformly, then the length is drawn uniformly in
    ``[min_interval_length, min(available, max)]`` where ``max`` is
    ``max_interval_prop * m`` (DrCIF uses 0.5). Dilation is fixed at 1, as in
    DrCIF. Returns a list of ``n`` (start, end) tuples (duplicates allowed, as
    the shared forest tolerates repeated columns).
    """
    min_l = min_interval_length
    max_l = max(min_l, int(max_interval_prop * m))
    intervals = []
    for _ in range(n):
        if rng.random() < 0.5:
            start = rng.randint(0, m + 1 - min_l) if m > min_l else 0
            len_range = min(m + 1 - start, max_l)
            length = (
                rng.randint(0, len_range - min_l) + min_l
                if len_range > min_l
                else min_l
            )
            end = start + length
        else:
            end = (
                rng.randint(0, m + 1 - min_l) + min_l if m > min_l else min_l
            )
            len_range = min(end, max_l)
            length = (
                rng.randint(0, len_range - min_l) + min_l
                if len_range > min_l
                else min_l
            )
            start = end - length
        intervals.append((start, end))
    return intervals


class SharedIntervalTransform:
    """Fixed-interval DrCIF feature pool transform, computed once and shared.

    Parameters
    ----------
    features : "drcif29" or "union35", default="drcif29"
        Per-interval attribute pool: the 29 DrCIF attributes (22 catch22 + 7
        summary stats), or those plus the six quantile features.
    interval_scheme : "dyadic" or "random", default="dyadic"
        "dyadic" uses the deterministic QUANT-style grid. "random" draws the
        same number of intervals per representation as the dyadic grid would
        contain, using DrCIF's interval generation rule (50/50 start/end
        anchor, length in [min, 0.5 * m]), seeded by random_state.
    min_interval_length : int, default=3
        Minimum interval length for both schemes.
    max_depth : int, default=6
        Maximum dyadic depth (also sets the interval count for "random").
    max_interval_prop : float, default=0.5
        Maximum interval length as a proportion of the series length, used by
        the "random" scheme (DrCIF's default is 0.5).
    random_state : int, RandomState instance or None, default=None
        Only used by the "random" interval scheme.
    """

    def __init__(
        self,
        features="drcif29",
        interval_scheme="dyadic",
        min_interval_length=3,
        max_depth=6,
        max_interval_prop=0.5,
        random_state=None,
    ):
        if features not in ("drcif29", "union35"):
            raise ValueError(f"Unknown features input: {features}")
        if interval_scheme not in ("dyadic", "random"):
            raise ValueError(f"Unknown interval_scheme input: {interval_scheme}")
        self.features = features
        self.interval_scheme = interval_scheme
        self.min_interval_length = min_interval_length
        self.max_depth = max_depth
        self.max_interval_prop = max_interval_prop
        self.random_state = random_state

        self._stat_funcs = (
            _STAT_FUNCS_29 if features == "drcif29" else _STAT_FUNCS_35
        )
        self._catch22 = Catch22(outlier_norm=True)
        self._periodogram = PeriodogramTransformer()

    def fit(self, X):
        """Fix the interval sets from the training data shape."""
        reps = self._representations(X, fit_periodogram=True)
        rng = check_random_state(self.random_state)

        self.intervals_ = []
        for rep in reps:
            m = rep.shape[2]
            dyadic = dyadic_intervals(m, self.min_interval_length, self.max_depth)
            if self.interval_scheme == "dyadic":
                self.intervals_.append(dyadic)
            else:
                # DrCIF's random interval model, matched in count to the dyadic
                # grid so the only variable versus the dyadic scheme is how
                # interval positions and lengths are chosen
                self.intervals_.append(
                    drcif_random_intervals(
                        m,
                        len(dyadic),
                        rng,
                        self.min_interval_length,
                        self.max_interval_prop,
                    )
                )

        self.n_features_ = sum(
            len(ivs) * (22 + len(self._stat_funcs)) * X.shape[1]
            for ivs in self.intervals_
        )
        return self

    def transform(self, X):
        """Compute the shared feature matrix, float32, NaN/inf replaced by 0."""
        reps = self._representations(X, fit_periodogram=False)

        cols = []
        for rep, intervals in zip(reps, self.intervals_):
            for s, e in intervals:
                sl = rep[:, :, s:e]
                cols.append(self._catch22.fit_transform(sl))
                for ch in range(rep.shape[1]):
                    rows = np.ascontiguousarray(sl[:, ch, :])
                    for f in self._stat_funcs:
                        cols.append(f(rows).reshape(-1, 1))

        Xt = np.hstack(cols).astype(np.float32)
        return np.nan_to_num(Xt, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _representations(self, X, fit_periodogram):
        X = np.asarray(X, dtype=np.float64)
        if fit_periodogram:
            per = self._periodogram.fit_transform(X)
        else:
            per = self._periodogram.transform(X)
        return [X, first_order_differences_3d(X), np.asarray(per)]
