"""Functions to run algorithm scalability experiments."""

import time

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
    make_example_3d_numpy,
)
from sklearn.utils import check_random_state


def run_timing_experiment(
    estimators,
    input_type="collection",
    dimension="n_timepoints",
    function="fit",
    random_state=None,
):
    """Return the time taken to run estimator functions for randomly generated data.

    Will time the function for each estimator in milliseconds, gradually increasing the
    size of the chosen dimension. The time taken will be stored in a dictionary.

    Parameters
    ----------
    estimators : list
        List of estimators to be evaluated.
    input_type : str, default="collection"
        Type of input data to be generated. Options are "collection" or "series".
    dimension : str, default="n_timepoints"
        Type of scaler to be used. Options are "n_cases", "n_channels", or
        "n_timepoints".

        "n_cases" is only valid for input_type="collection".
    function : str, default="fit"
        Function to be timed. Options are "fit", "predict", "fit_predict",
        "predict_proba", "fit_predict_proba", "transform", or "fit_transform".

        For "predict", "predict_proba" and "transform" the function will be timed
        after the estimator has been fitted.
    random_state : int or None, default=None
        Random state to be used for data generation and estimator cloning.

    Returns
    -------
    dict
        Dictionary of timings for each estimator and data size pair.
    """
    timings = {}
    rng = check_random_state(random_state)

    for i in range(1, 11):
        if input_type == "collection":
            if dimension == "n_cases":
                size = 50 * i
                X, y = make_example_3d_numpy(
                    n_cases=size,
                    n_channels=1,
                    n_timepoints=100,
                    random_state=rng.randint(np.iinfo(np.int32).max),
                )
            elif dimension == "n_channels":
                size = i
                X, y = make_example_3d_numpy(
                    n_cases=50,
                    n_channels=size,
                    n_timepoints=100,
                    random_state=rng.randint(np.iinfo(np.int32).max),
                )
            elif dimension == "n_timepoints":
                size = 100 * i
                X, y = make_example_3d_numpy(
                    n_cases=50,
                    n_channels=1,
                    n_timepoints=size,
                    random_state=rng.randint(np.iinfo(np.int32).max),
                )
            else:
                raise ValueError(f"Invalid dimension {dimension}")
        elif input_type == "series":
            if dimension == "n_channels":
                size = i
                X = make_example_2d_numpy_series(
                    n_channels=size,
                    n_timepoints=100,
                    random_state=rng.randint(np.iinfo(np.int32).max),
                )
                y = None
            elif dimension == "n_timepoints":
                size = 100 * i
                X = make_example_2d_numpy_series(
                    n_channels=1,
                    n_timepoints=size,
                    random_state=rng.randint(np.iinfo(np.int32).max),
                )
                y = None
            else:
                raise ValueError(f"Invalid dimension {dimension}")

        for estimator in estimators:
            estimator = _clone_estimator(
                estimator, random_state=rng.randint(np.iinfo(np.int32).max)
            )
            key = (estimator.__class__.__name__, size)

            if function == "fit":
                timings[key] = _time_function(estimator.fit, 10, X, y)
            elif function == "predict":
                estimator.fit(X, y)
                timings[key] = _time_function(estimator.predict, 10, X)
            elif function == "fit_predict":
                timings[key] = _time_function(estimator.fit_predict, 10, X, y)
            elif function == "predict_proba":
                estimator.fit(X, y)
                timings[key] = _time_function(estimator.predict_proba, 10, X)
            elif function == "fit_predict_proba":
                timings[key] = _time_function(estimator.fit_predict_proba, 10, X, y)
            elif function == "transform":
                estimator.fit(X, y)
                timings[key] = _time_function(estimator.transform, 10, X)
            elif function == "fit_transform":
                timings[key] = _time_function(estimator.fit_transform, 10, X, y)
            else:
                raise ValueError(f"Invalid function {function}")

    return timings


def _time_function(function, loops, *args):
    """Time a function and return the time taken."""
    t = 0
    for i in range(loops + 1):
        start = time.time()
        function(*args)
        if i != 0:
            t += (time.time() - start) * 1000
    return t / loops
