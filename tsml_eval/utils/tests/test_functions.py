"""Test utility functions."""

import numpy as np
import pytest

from tsml_eval.utils.functions import (
    pair_list_to_dict,
    rank_array,
    time_to_milliseconds,
)


def test_pair_list_to_dict():
    """Test pair_list_to_dict function."""
    assert pair_list_to_dict([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}
    assert pair_list_to_dict(None) == {}


def test_rank_array():
    """Test array ranking function."""
    arr1 = [
        0.611111111,
        0.638888889,
        0.666666667,
        0.666666667,
        0.611111111,
        0.666666667,
        0.611111111,
        0.638888889,
        0.666666667,
        0.666666667,
        0.666666667,
    ]
    arr2 = [
        0.683333333,
        0.7,
        0.716666667,
        0.666666667,
        0.783333333,
        0.516666667,
        0.4,
        0.583333333,
        0.633333333,
        0.533333333,
        0.583333333,
    ]
    arr3 = [0.584, 0.6, 0.604, 0.548, 0.616, 0.504, 0.584, 0.588, 0.544, 0.572, 0.516]
    arr4 = [
        0.342541436,
        0.370165746,
        0.364640884,
        0.375690608,
        0.46961326,
        0.337016575,
        0.359116022,
        0.453038674,
        0.419889503,
        0.303867403,
        0.29281768,
    ]
    ranks1 = [10, 7.5, 3.5, 3.5, 10, 3.5, 10, 7.5, 3.5, 3.5, 3.5]
    ranks2 = [4, 3, 2, 5, 1, 10, 11, 7.5, 6, 9, 7.5]
    ranks3 = [5.5, 3, 2, 8, 1, 11, 5.5, 4, 9, 7, 10]
    ranks4 = [8, 5, 6, 4, 1, 9, 7, 2, 3, 10, 11]

    assert (rank_array(np.array(arr1)) == np.array(ranks1)).all()
    assert (rank_array(np.array(arr2)) == np.array(ranks2)).all()
    assert (rank_array(np.array(arr3)) == np.array(ranks3)).all()
    assert (rank_array(np.array(arr4)) == np.array(ranks4)).all()

    inverse_ranks1 = [2, 4.5, 8.5, 8.5, 2, 8.5, 2, 4.5, 8.5, 8.5, 8.5]
    inverse_ranks2 = [8, 9, 10, 7, 11, 2, 1, 4.5, 6, 3, 4.5]
    inverse_ranks3 = [6.5, 9, 10, 4, 11, 1, 6.5, 8, 3, 5, 2]
    inverse_ranks4 = [4, 7, 6, 8, 11, 3, 5, 10, 9, 2, 1]

    assert (
        rank_array(np.array(arr1), higher_better=False) == np.array(inverse_ranks1)
    ).all()
    assert (
        rank_array(np.array(arr2), higher_better=False) == np.array(inverse_ranks2)
    ).all()
    assert (
        rank_array(np.array(arr3), higher_better=False) == np.array(inverse_ranks3)
    ).all()
    assert (
        rank_array(np.array(arr4), higher_better=False) == np.array(inverse_ranks4)
    ).all()


def test_time_to_milliseconds_invalid_unit():
    """Test time_to_milliseconds function with invalid time unit."""
    with pytest.raises(ValueError, match="Unknown time unit"):
        time_to_milliseconds(500, "gigaseconds")
