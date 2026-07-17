"""Row-wise quantile features for the NewDrCIF attribute pool.

Each function follows the contract of the row_* functions in
aeon.utils.numba.stats: input is a 2D array of interval slices (one row per
case), output is a 1D array of one value per case. Q0/Q50/Q100 are excluded as
they duplicate the min/median/max summary stats already in the pool.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "row_quantile_10",
    "row_quantile_25",
    "row_quantile_75",
    "row_quantile_90",
    "row_quantile_25_centred",
    "row_quantile_75_centred",
]

import numpy as np
from numba import njit


@njit(cache=True)
def _quantile(x, q):
    # linear interpolation on sorted values, matching np.quantile's default
    # method, so values agree with numpy
    s = np.sort(x)
    pos = q * (s.shape[0] - 1)
    lo = int(pos)
    frac = pos - lo
    if frac == 0.0:
        return s[lo]
    return s[lo] + frac * (s[lo + 1] - s[lo])


@njit(cache=True)
def _row_quantile(X, q, centre):
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        out[i] = _quantile(X[i], q)
        if centre:
            out[i] -= np.mean(X[i])
    return out


@njit(cache=True)
def row_quantile_10(X):
    """Q10 of each row."""
    return _row_quantile(X, 0.10, False)


@njit(cache=True)
def row_quantile_25(X):
    """Q25 of each row."""
    return _row_quantile(X, 0.25, False)


@njit(cache=True)
def row_quantile_75(X):
    """Q75 of each row."""
    return _row_quantile(X, 0.75, False)


@njit(cache=True)
def row_quantile_90(X):
    """Q90 of each row."""
    return _row_quantile(X, 0.90, False)


@njit(cache=True)
def row_quantile_25_centred(X):
    """Q25 minus the mean of each row."""
    return _row_quantile(X, 0.25, True)


@njit(cache=True)
def row_quantile_75_centred(X):
    """Q75 minus the mean of each row."""
    return _row_quantile(X, 0.75, True)
