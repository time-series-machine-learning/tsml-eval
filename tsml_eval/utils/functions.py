"""Miscellaneous functions for tsml_eval."""

__all__ = [
    "str_in_nested_list",
    "pair_list_to_dict",
    "time_to_milliseconds",
    "rank_array",
]

import numpy as np


def str_in_nested_list(nested_list, item):
    """Find an item in a nested list."""
    if item in (s.casefold() for s in nested_list if isinstance(s, str)):
        return True
    else:
        return any(
            str_in_nested_list(nl, item) for nl in nested_list if isinstance(nl, list)
        )


def pair_list_to_dict(pl):
    """Convert a 2d list of pairs to a dict.

    Each list item must be a tuple or list of length 2. The first item in each pair
    is used as the key, the second as the value.

    If ls is None, returns an empty dict.
    """
    return {} if pl is None else {k: v for k, v in pl}


def time_to_milliseconds(time_value, time_unit):
    """Convert a time value from the given time unit to milliseconds.

    Parameters
    ----------
    time_value : float
        The time value to convert.
    time_unit : str {"nanoseconds", "microseconds", "milliseconds", "seconds",
                    "minutes", "hours", "days"}
        The current time unit of the value.

    Returns
    -------
    float
        The time in milliseconds.
    """
    time_unit = time_unit.lower().strip()
    time_units = {
        "nanoseconds": 1e-6,
        "microseconds": 1e-3,
        "milliseconds": 1,
        "seconds": 1e3,
        "minutes": 60e3,
        "hours": 3600e3,
        "days": 86400e3,
    }

    if time_unit not in time_units:
        raise ValueError(f"Unknown time unit: {time_unit}")

    # Convert the time value to milliseconds
    return time_value * time_units[time_unit]


def rank_array(arr, higher_better=True):
    """
    Assign a rank to each value in a 1D numpy array.

    A lower rank number is assumed to be better. Lower values can receive better ranks
    or vice versa based on the `higher_better` parameter. Equal values receive the
    average of the ranks they would cover.

    Parameters
    ----------
    arr : numpy.ndarray
        The input 1D array containing values to be ranked.
    higher_better : bool, default=True
        If True, lower values receive better ranks.
        If False (default), higher values receive better ranks.

    Returns
    -------
    ranks : numpy.ndarray
        Array of ranks, same shape as `arr`.
    """
    # argsort returns indices that would sort the array
    sorter = np.argsort(arr)
    ranks = np.zeros(len(arr), dtype=float)
    ranks[sorter] = np.arange(1, len(arr) + 1)

    # Handle ties: find unique values and their corresponding indices
    unique_vals, inv_sorter = np.unique(arr, return_inverse=True)
    for i in np.unique(inv_sorter):
        ranks[inv_sorter == i] = np.mean(ranks[inv_sorter == i])

    if higher_better:
        ranks = len(arr) + 1 - ranks

    return ranks
