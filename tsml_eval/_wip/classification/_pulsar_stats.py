"""Two cheap order-aware local statistics ported from PULSAR.

Both follow the row_* contract of aeon.utils.numba.stats: input is a 2D array of
interval slices (one row per case), output is a 1D array of one value per case.
mean_crossing is order-sensitive (counts transitions across the mean); values
above mean is order-invariant. PULSAR reaches SOTA using only simple stats like
these, which motivated adding them to the pool.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["row_mean_crossing", "row_prop_above_mean"]

import numpy as np
from numba import njit


@njit(cache=True)
def row_mean_crossing(X):
    """Proportion of successive pairs that cross the row mean (normalised)."""
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        row = X[i]
        n = row.shape[0]
        if n <= 1:
            continue
        m = np.mean(row)
        count = 0
        for j in range(n - 1):
            prev = row[j]
            cur = row[j + 1]
            if (prev <= m and cur > m) or (prev >= m and cur < m):
                count += 1
        out[i] = count / (n - 1)
    return out


@njit(cache=True)
def row_prop_above_mean(X):
    """Proportion of values strictly above the row mean."""
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        row = X[i]
        m = np.mean(row)
        count = 0
        for j in range(row.shape[0]):
            if row[j] > m:
                count += 1
        out[i] = count / row.shape[0]
    return out
