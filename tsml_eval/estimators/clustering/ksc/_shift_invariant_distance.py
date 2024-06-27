"""Shift-invariant distance."""

from typing import List, Optional, Union

import numpy as np
from aeon.distances._utils import _convert_to_list, _is_multivariate
from numba import njit
from numba.typed import List as NumbaList


@njit(cache=True, fastmath=True)
def shift_invariant_distance(x: np.ndarray, y: np.ndarray, max_shift: int = 2) -> float:
    if x.ndim == 1 and y.ndim == 1:
        dist, _ = _univariate_shift_invariant_distance(x, y, max_shift)
        return dist
    if x.ndim == 2 and y.ndim == 2:
        if x.shape[0] == 1 and y.shape[0] == 1:
            dist, _ = _univariate_shift_invariant_distance(x[0, :], y[0, :], max_shift)
            return dist
        else:
            n_channels = min(x.shape[0], y.shape[0])
            distance = 0.0
            for i in range(n_channels):
                dist, _ = _univariate_shift_invariant_distance(x[i], y[i], max_shift)
                distance += dist

            return distance

    raise ValueError("x and y must be 1D or 2D")


def shift_invariant_pairwise_distance(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    max_shift: int = 2,
) -> np.ndarray:
    multivariate_conversion = _is_multivariate(X, y)
    _X, _ = _convert_to_list(X, "", multivariate_conversion)

    if y is None:
        return _shift_invariant_pairwise_distance_single(_X, max_shift)

    _y, _ = _convert_to_list(y, "y", multivariate_conversion)
    return _shift_invariant_pairwise_distance(_X, _y, max_shift)


# @njit(cache=True, fastmath=True)
def shift_invariant_best_shift(
    x: np.ndarray, y: np.ndarray, max_shift: int = 2
) -> Union[float, np.ndarray]:
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_shift_invariant_distance(x, y, max_shift)
    if x.ndim == 2 and y.ndim == 2:
        if x.shape[0] == 1 and y.shape[0] == 1:
            return _univariate_shift_invariant_distance(x[0, :], y[0, :], max_shift)
        else:
            n_channels = min(x.shape[0], y.shape[0])
            distance = 0.0
            best_shift = np.zeros((n_channels, y.shape[1]))
            for i in range(n_channels):
                dist, curr_shift = _univariate_shift_invariant_distance(
                    x[i], y[i], max_shift
                )
                best_shift[i] = curr_shift
                distance += dist

            return distance, best_shift

    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _shift_invariant_pairwise_distance_single(
    x: NumbaList[np.ndarray], max_shift: int
) -> np.ndarray:
    n_cases = len(x)
    distances = np.zeros((n_cases, n_cases))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = shift_invariant_distance(x[i], x[j], max_shift)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _shift_invariant_pairwise_distance(
    x: NumbaList[np.ndarray], y: NumbaList[np.ndarray], max_shift: int
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = shift_invariant_distance(x[i], y[j], max_shift)
    return distances


# Constant which cant be computed during numba compilation but can be passed as global
SMALLEST_FLOAT = np.finfo(float).eps


@njit(cache=True, fastmath=True)
def _scale_distance(x, y, n_instances):
    numerator = 0.0
    denominator = 0.0
    for i in range(n_instances):
        numerator += x[i] * y[i]
        denominator += y[i] * y[i]
    denominator += SMALLEST_FLOAT
    alpha = numerator / denominator

    # Calculate the distance using a raw loop
    sum_diff_squared = 0.0
    sum_x_squared = 0.0
    for i in range(n_instances):
        diff = x[i] - y[i] * alpha
        sum_diff_squared += diff * diff
        sum_x_squared += x[i] * x[i]
    return np.sqrt(sum_diff_squared) / (np.sqrt(sum_x_squared) + SMALLEST_FLOAT)


@njit(cache=True, fastmath=True)
def _univariate_shift_invariant_distance(
    x: np.ndarray, y: np.ndarray, max_shift: int
) -> Union[float, np.ndarray]:
    n_timepoints = min(len(x), len(y))
    min_dist = _scale_distance(x, y, n_timepoints)
    best_shift = y
    for sh in range(-max_shift, max_shift + 1):
        if sh == 0:
            continue
        elif sh < 0:
            shifted_y = np.append(y[-sh:], np.zeros(-sh))
        else:
            shifted_y = np.append(np.zeros(sh), y[:-sh])

        dist = _scale_distance(x, shifted_y, n_timepoints)

        if dist < min_dist:
            min_dist = dist
            best_shift = shifted_y

    return min_dist, best_shift
