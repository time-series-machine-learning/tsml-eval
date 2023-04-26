# -*- coding: utf-8 -*-
import time
import warnings

from aeon.distances import distance_factory
from tslearn.metrics import dtw as tslearn_dtw
from dtw import dtw as dtw_python_dtw
from rust_dtw import dtw as rust_dtw_dtw
from aeon.distances.tests._utils import create_test_distance_numpy

import pandas as pd

warnings.filterwarnings(
    "ignore"
)  # Hide warnings that can generate and clutter notebook


def timing_experiment(x, y, distance_callable, distance_params=None, average=200):
    """Time the average time it takes to take the distance from the first time series
    to all of the other time series in X.

    Parameters
    ----------
    X: np.ndarray
        A dataset of time series.
    distance_callable: Callable
        A callable that is the distance function to time.

    Returns
    -------
    float
        Average time it took to run a distance
    """
    if distance_params is None:
        distance_params = {}
    total_time = 0
    for i in range(0, average):
        start = time.time()
        curr_dist = distance_callable(x, y, **distance_params)
        total_time += time.time() - start

    return total_time


def univariate_experiment(start=1000, end=10000, increment=1000):
    aeon_timing = []
    tslearn_timing = []
    dtw_python_timing = []
    rust_dtw_timing = []

    col_headers = []

    for i in range(start, end + increment, increment):
        col_headers.append(i)
        distance_m_d = create_test_distance_numpy(2, 1, i)

        x = distance_m_d[0][0]
        y = distance_m_d[1][0]
        numba_aeon = distance_factory(x, y, metric="dtw")
        print(f" length = {i} )")
        aeon_time = timing_experiment(distance_m_d[0], distance_m_d[1], numba_aeon)
        print(f" aeon = {aeon_time} )")
        tslearn_time = timing_experiment(x, y, tslearn_dtw)
        print(f" tslearn = {tslearn_time} )")
        rust_dtw_time = timing_experiment(
            x, y, rust_dtw_dtw, {"window": i, "distance_mode": "euclidean"}
        )
        print(f" rust = {rust_dtw_time} )")
        dtw_python_time = timing_experiment(x, y, dtw_python_dtw)
        print(f" dtw_python_time = {dtw_python_time} )")
        aeon_timing.append(aeon_time)
        tslearn_timing.append(tslearn_time)
        dtw_python_timing.append(dtw_python_time)
        rust_dtw_timing.append(rust_dtw_time)

    uni_df = pd.DataFrame(
        {
            "time points": col_headers,
            "aeon": aeon_timing,
            "tslearn": tslearn_timing,
            "rust-dtw": rust_dtw_timing,
            "dtw-python": dtw_python_timing,
        }
    )
    return uni_df


def multivariate_experiment(start=100, end=500, increment=100):
    aeon_timing = []
    tslearn_timing = []

    col_headers = []

    for i in range(start, end + increment, increment):
        col_headers.append(i)
        distance_m_d = create_test_distance_numpy(2, i, i)

        x = distance_m_d[0]
        y = distance_m_d[1]
        tslearn_x = x.reshape((x.shape[1], x.shape[0]))  # tslearn wants m, d format
        tslearn_y = y.reshape((y.shape[1], y.shape[0]))  # tslearn wants m, d format
        numba_aeon = distance_factory(x, y, metric="dtw")

        tslearn_time = timing_experiment(tslearn_x, tslearn_y, tslearn_dtw)
        aeon_time = timing_experiment(x, y, numba_aeon)

        aeon_timing.append(aeon_time)
        tslearn_timing.append(tslearn_time)

    multi_df = pd.DataFrame(
        {
            "time points": col_headers,
            "aeon": aeon_timing,
            "tslearn": tslearn_timing,
        }
    )
    return multi_df


if __name__ == "__main__":
    uni_df = univariate_experiment(start=7000, end=10000, increment=1000)
    uni_df.to_csv("./uni_dist_results", index=False)

#    multi_df = multivariate_experiment(
#       start=100,
#      end=200,
#      increment=100
#  )
# multi_df.to_csv('./multi_dist_results', index=False)
