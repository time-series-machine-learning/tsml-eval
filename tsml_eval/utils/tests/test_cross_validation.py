"""Tests for dependent time-series cross-validation utilities."""

import numpy as np
import pytest
from sklearn import config_context
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import BaseCrossValidator, cross_val_score

from tsml_eval.utils.cross_validation import (
    BufferedBlockedKFold,
    HVBlockCrossValidator,
)


def _assert_split_equal(split, expected_train, expected_test):
    """Assert that a generated split contains the expected indices."""
    train, test = split
    np.testing.assert_array_equal(train, expected_train)
    np.testing.assert_array_equal(test, expected_test)


def test_hv_block_v_zero_expected_indices():
    """Test exact h-block indices when v=0, including sequence boundaries."""
    splits = list(HVBlockCrossValidator(h=1, v=0).split(np.arange(5)))

    expected = [
        ([2, 3, 4], [0]),
        ([3, 4], [1]),
        ([0, 4], [2]),
        ([0, 1], [3]),
        ([0, 1, 2], [4]),
    ]
    assert len(splits) == len(expected)
    for split, (train, test) in zip(splits, expected):
        _assert_split_equal(split, train, test)


def test_hv_block_h_and_v_zero_is_leave_one_out():
    """Test that h=0 and v=0 give leave-one-out cross-validation."""
    n_cases = 6
    splits = list(HVBlockCrossValidator(h=0, v=0).split(np.arange(n_cases)))

    np.testing.assert_array_equal(
        np.concatenate([test for _, test in splits]), np.arange(n_cases)
    )
    for train, test in splits:
        assert len(train) == n_cases - 1
        assert np.intersect1d(train, test).size == 0


def test_hv_block_v_zero_groups_limit_buffering_to_each_sequence():
    """Test that neighbours across an independent sequence boundary remain."""
    X = np.arange(6)
    groups = np.repeat(["sequence_a", "sequence_b"], 3)
    splits = list(HVBlockCrossValidator(h=1, v=0).split(X, groups=groups))

    _assert_split_equal(splits[2], [0, 3, 4, 5], [2])
    _assert_split_equal(splits[3], [0, 1, 2, 5], [3])


def test_hv_block_group_order_is_input_order():
    """Test that non-contiguous group members retain their input ordering."""
    X = np.arange(8)
    groups = np.tile([0, 1], 4)
    splits = list(HVBlockCrossValidator(h=1, v=0).split(X, groups=groups))

    # Within group 0 the neighbours of case 2 are cases 0 and 4.
    _assert_split_equal(splits[2], [1, 3, 5, 6, 7], [2])


def test_hv_block_v_zero_sklearn_interface_and_determinism():
    """Test BaseCrossValidator compatibility and repeatable splits."""
    X = np.arange(8)
    cv = HVBlockCrossValidator(h=np.int64(1), v=0)

    assert isinstance(cv, BaseCrossValidator)
    assert cv.get_n_splits(X) == len(X)
    first = list(cv.split(X))
    second = list(cv.split(X))
    for first_split, second_split in zip(first, second):
        _assert_split_equal(first_split, *second_split)


@pytest.mark.parametrize("h", [-1, 1.5, True, "1"])
def test_hv_block_invalid_h(h):
    """Test that h must be a non-negative integer."""
    with pytest.raises(ValueError, match="h must be a non-negative integer"):
        HVBlockCrossValidator(h=h)


def test_hv_block_expected_sliding_indices():
    """Test the sliding, overlapping test blocks defined for hv-block CV."""
    splits = list(HVBlockCrossValidator(h=1, v=1).split(np.arange(8)))

    expected = [
        ([4, 5, 6, 7], [0, 1, 2]),
        ([5, 6, 7], [1, 2, 3]),
        ([0, 6, 7], [2, 3, 4]),
        ([0, 1, 7], [3, 4, 5]),
        ([0, 1, 2], [4, 5, 6]),
        ([0, 1, 2, 3], [5, 6, 7]),
    ]
    assert len(splits) == len(expected)
    for split, (train, test) in zip(splits, expected):
        _assert_split_equal(split, train, test)


def test_hv_block_groups_apply_sliding_windows_per_sequence():
    """Test that each independent sequence supplies its own sliding windows."""
    X = np.arange(9)
    groups = np.repeat(["sequence_a", "sequence_b"], [5, 4])
    cv = HVBlockCrossValidator(h=1, v=1)
    splits = list(cv.split(X, groups=groups))

    assert cv.get_n_splits(X, groups=groups) == 5
    _assert_split_equal(splits[0], [4, 5, 6, 7, 8], [0, 1, 2])
    _assert_split_equal(splits[2], [0, 5, 6, 7, 8], [2, 3, 4])
    _assert_split_equal(splits[3], [0, 1, 2, 3, 4], [5, 6, 7])


def test_hv_block_sklearn_interface_and_regression_targets():
    """Test fold counting, BaseCrossValidator compatibility, and ignored targets."""
    X = np.arange(10)
    y = np.linspace(0, 1, len(X))
    cv = HVBlockCrossValidator(h=np.int64(1), v=np.int64(2))

    assert isinstance(cv, BaseCrossValidator)
    assert cv.get_n_splits(X) == 6
    with_y = list(cv.split(X, y))
    without_y = list(cv.split(X))
    for first_split, second_split in zip(with_y, without_y):
        _assert_split_equal(first_split, *second_split)


@pytest.mark.parametrize("v", [-1, 1.5, True, "1"])
def test_hv_block_invalid_v(v):
    """Test that v must be a non-negative integer."""
    with pytest.raises(ValueError, match="v must be a non-negative integer"):
        HVBlockCrossValidator(v=v)


def test_hv_block_invalid_data():
    """Test invalid hv-block data and configurations."""
    cv = HVBlockCrossValidator(v=1)
    with pytest.raises(TypeError, match="required positional argument: 'X'"):
        cv.get_n_splits()
    with pytest.raises(ValueError, match="at least one case"):
        cv.get_n_splits(np.arange(0))
    with pytest.raises(ValueError, match=r"at least 2 \* v \+ 1 cases"):
        cv.get_n_splits(np.arange(2))
    with pytest.raises(ValueError, match="at least 2 cases"):
        next(HVBlockCrossValidator().split(np.arange(1)))
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        next(cv.split(np.arange(4), groups=[0, 0, 0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        next(cv.split(np.arange(4), groups=np.zeros((4, 1))))
    with pytest.raises(ValueError, match=r"at least 2 \* v \+ 1 cases"):
        next(cv.split(np.arange(5), groups=[0, 0, 1, 1, 1]))
    with pytest.raises(ValueError, match="Reduce h or v"):
        next(HVBlockCrossValidator(h=2, v=1).split(np.arange(7)))
    with pytest.raises(ValueError, match="Reduce h"):
        next(HVBlockCrossValidator(h=2, v=0).split(np.arange(5)))


def test_buffered_blocked_kfold_expected_indices():
    """Test exact partitioned block indices and two-sided buffering."""
    splits = list(BufferedBlockedKFold(n_splits=3, gap=1).split(np.arange(10)))

    expected = [
        ([5, 6, 7, 8, 9], [0, 1, 2, 3]),
        ([0, 1, 2, 8, 9], [4, 5, 6]),
        ([0, 1, 2, 3, 4, 5], [7, 8, 9]),
    ]
    assert len(splits) == len(expected)
    for split, (train, test) in zip(splits, expected):
        _assert_split_equal(split, train, test)


def test_buffered_blocked_kfold_grouped_unequal_sequences():
    """Test paired folds for independent sequences of unequal length."""
    X = np.arange(9)
    groups = np.repeat(["sequence_a", "sequence_b"], [5, 4])
    y = np.repeat([0, 1], [5, 4])
    splits = list(BufferedBlockedKFold(n_splits=2, gap=1).split(X, y, groups))

    _assert_split_equal(splits[0], [4, 8], [0, 1, 2, 5, 6])
    _assert_split_equal(splits[1], [0, 1, 5], [3, 4, 7, 8])
    for train, test in splits:
        assert set(y[test]) == {0, 1}
        assert np.intersect1d(train, test).size == 0


@pytest.mark.parametrize("gap", [0, 1, 2])
def test_buffered_blocked_kfold_complete_coverage_and_determinism(gap):
    """Test that partition mode covers every case once and is deterministic."""
    X = np.arange(17)
    groups = np.repeat([0, 1, 2], [7, 6, 4])
    cv = BufferedBlockedKFold(n_splits=4, gap=gap)

    first = list(cv.split(X, groups=groups))
    second = list(cv.split(X, groups=groups))
    test_indices = np.concatenate([test for _, test in first])

    np.testing.assert_array_equal(np.sort(test_indices), np.arange(len(X)))
    assert len(np.unique(test_indices)) == len(X)
    for first_split, second_split in zip(first, second):
        _assert_split_equal(first_split, *second_split)


def test_buffered_blocked_kfold_regression_targets_are_accepted():
    """Test that continuous targets do not affect partitioned block splits."""
    X = np.arange(12)
    y = np.linspace(0, 1, len(X))
    cv = BufferedBlockedKFold(n_splits=3, gap=1)

    with_y = list(cv.split(X, y))
    without_y = list(cv.split(X))
    for first_split, second_split in zip(with_y, without_y):
        _assert_split_equal(first_split, *second_split)


def test_buffered_blocked_kfold_sklearn_interface():
    """Test the BaseCrossValidator interface."""
    cv = BufferedBlockedKFold(n_splits=np.int64(3), gap=np.int64(1))

    assert isinstance(cv, BaseCrossValidator)
    assert cv.get_n_splits() == 3


@pytest.mark.parametrize(
    "cv",
    [
        HVBlockCrossValidator(h=1, v=1),
        BufferedBlockedKFold(n_splits=3, gap=1),
    ],
)
def test_sklearn_metadata_routing(cv):
    """Test that sklearn metadata routing passes groups to every splitter."""
    X = np.arange(12).reshape(-1, 1)
    y = np.arange(12, dtype=float)
    groups = np.repeat([0, 1], 6)
    with config_context(enable_metadata_routing=True):
        scores = cross_val_score(
            DummyRegressor(),
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
            params={"groups": groups},
        )

    assert len(scores) == cv.get_n_splits(X, y, groups)


@pytest.mark.parametrize("n_splits", [-1, 0, 1, 1.5, True, "3"])
def test_buffered_blocked_kfold_invalid_n_splits(n_splits):
    """Test that n_splits must be an integer of at least two."""
    with pytest.raises(ValueError, match="n_splits must be an integer"):
        BufferedBlockedKFold(n_splits=n_splits)


@pytest.mark.parametrize("gap", [-1, 1.5, True, "1"])
def test_buffered_blocked_kfold_invalid_gap(gap):
    """Test that gap must be a non-negative integer."""
    with pytest.raises(ValueError, match="gap must be a non-negative integer"):
        BufferedBlockedKFold(gap=gap)


def test_buffered_blocked_kfold_invalid_data():
    """Test invalid grouped partition data and configurations."""
    cv = BufferedBlockedKFold(n_splits=3)
    with pytest.raises(ValueError, match="smallest sequence size of 2"):
        next(cv.split(np.arange(5), groups=[0, 0, 1, 1, 1]))
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        next(cv.split(np.arange(6), groups=[0, 0, 0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        next(cv.split(np.arange(6), groups=np.zeros((6, 1))))
    with pytest.raises(ValueError, match="at least one case"):
        next(cv.split(np.arange(0)))
    with pytest.raises(ValueError, match="increase n_splits"):
        next(BufferedBlockedKFold(n_splits=3, gap=4).split(np.arange(10)))
