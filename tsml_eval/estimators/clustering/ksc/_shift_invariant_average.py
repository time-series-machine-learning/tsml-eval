"""Shift-invariant average."""

import numpy as np
from numpy.linalg import eigh, norm

from tsml_eval.estimators.clustering.ksc._shift_invariant_distance import (
    shift_invariant_best_shift,
)


def shift_invariant_average(
    X: np.ndarray, initial_center: np.ndarray, max_shift: int = 2
):
    optimal_shifts = np.zeros_like(X)

    for i in range(X.shape[0]):
        if not initial_center.any():
            optimal_shifts[i] = X[i]
        else:
            _, curr_shift = shift_invariant_best_shift(initial_center, X[i], max_shift)
            optimal_shifts[i] = curr_shift

    new_center = np.zeros_like(initial_center)
    n_cases, n_dims, n_timepoints = X.shape

    if optimal_shifts.shape[0] == 0:
        return new_center

    for d_i in range(n_dims):
        a_di = optimal_shifts[:, d_i, :]
        b = a_di / np.tile(norm(a_di, axis=1), (n_timepoints, 1)).T
        M = np.matmul(b.T, b) - (n_cases + 1) * np.eye(n_timepoints)

        w, v = eigh(M)
        ksc_di = v[:, np.argmax(w)]

        dist1 = norm(a_di[0, :] - ksc_di.T)
        dist2 = norm(a_di[0, :] - (-ksc_di.T))

        if dist1 > dist2:
            ksc_di = -ksc_di

        if np.sum(ksc_di) < 0:
            ksc_di = -ksc_di

        new_center[d_i, :] = ksc_di

    return new_center
