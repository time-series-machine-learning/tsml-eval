"""Tests for the dependent time series cross-validation splitters."""

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score

from tsml_eval.utils.splitters import BufferedBlockedKFold, HVBlockCrossValidator


def _assert_splits_equal(splits, expected):
    assert len(splits) == len(expected)
    for (train, test), (expected_train, expected_test) in zip(splits, expected):
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(test, expected_test)


def _brute_force_train(groups, test, buffer):
    """Return the cases outside the buffer around their sequence's test cases."""
    train = []
    for case in range(len(groups)):
        sequence = np.flatnonzero(groups == groups[case])
        test_positions = np.searchsorted(sequence, np.intersect1d(test, sequence))
        position = np.searchsorted(sequence, case)
        if len(test_positions) == 0 or np.abs(test_positions - position).min() > buffer:
            train.append(case)
    return np.array(train)


@pytest.mark.parametrize(
    "h, v, n_cases, expected",
    [
        (
            1,
            0,
            5,
            [
                ([2, 3, 4], [0]),
                ([3, 4], [1]),
                ([0, 4], [2]),
                ([0, 1], [3]),
                ([0, 1, 2], [4]),
            ],
        ),
        (
            1,
            1,
            8,
            [
                ([4, 5, 6, 7], [0, 1, 2]),
                ([5, 6, 7], [1, 2, 3]),
                ([0, 6, 7], [2, 3, 4]),
                ([0, 1, 7], [3, 4, 5]),
                ([0, 1, 2], [4, 5, 6]),
                ([0, 1, 2, 3], [5, 6, 7]),
            ],
        ),
    ],
)
def test_hv_block_expected_indices(h, v, n_cases, expected):
    """Test exact h-block (v=0) and hv-block indices for one sequence."""
    splits = list(HVBlockCrossValidator(h=h, v=v).split(np.arange(n_cases)))
    _assert_splits_equal(splits, expected)


@pytest.mark.parametrize("h", [0, 1, 2])
@pytest.mark.parametrize("v", [0, 1, 2])
def test_hv_block_buffers(h, v):
    """Test that blocks slide within each sequence and exclude h cases each side."""
    # Interleaved sequences of 5 ("a") and 7 ("b") cases.
    groups = np.array(["b", "a"] * 5 + ["b", "b"])
    X = np.arange(len(groups))
    cv = HVBlockCrossValidator(h=h, v=v)

    splits = list(cv.split(X, groups=groups))
    assert len(splits) == cv.get_n_splits(X, groups=groups) == 12 - 4 * v
    assert len({tuple(test) for _, test in splits}) == len(splits)
    first_cases = [test[0] for _, test in splits]
    assert first_cases == sorted(first_cases)

    for train, test in splits:
        assert len(set(groups[test])) == 1
        sequence = np.flatnonzero(groups == groups[test[0]])
        positions = np.searchsorted(sequence, test)
        np.testing.assert_array_equal(
            positions, np.arange(positions[0], positions[0] + 2 * v + 1)
        )
        np.testing.assert_array_equal(train, _brute_force_train(groups, test, h))


def test_buffered_blocked_kfold_expected_indices():
    """Test exact partitioned block indices and two-sided buffering."""
    splits = list(BufferedBlockedKFold(n_splits=3, gap=1).split(np.arange(10)))
    expected = [
        ([5, 6, 7, 8, 9], [0, 1, 2, 3]),
        ([0, 1, 2, 8, 9], [4, 5, 6]),
        ([0, 1, 2, 3, 4, 5], [7, 8, 9]),
    ]
    _assert_splits_equal(splits, expected)


@pytest.mark.parametrize("n_splits", [2, 3, 4])
@pytest.mark.parametrize("gap", [0, 1, 2])
def test_buffered_blocked_kfold_buffers(n_splits, gap):
    """Test that every case is tested once and gap cases each side leave training."""
    # Interleaved sequences of 8 ("a"), 7 ("b") and 6 ("c") cases.
    groups = np.array(["b", "a", "c"] * 6 + ["a", "b", "a"])
    X = np.arange(len(groups))
    splits = list(
        BufferedBlockedKFold(n_splits=n_splits, gap=gap).split(X, groups=groups)
    )

    assert len(splits) == n_splits
    np.testing.assert_array_equal(np.sort(np.concatenate([t for _, t in splits])), X)

    for train, test in splits:
        for group in ["a", "b", "c"]:
            sequence = np.flatnonzero(groups == group)
            positions = np.searchsorted(sequence, np.intersect1d(test, sequence))
            assert len(positions) > 0
            np.testing.assert_array_equal(
                positions, np.arange(positions[0], positions[-1] + 1)
            )
        np.testing.assert_array_equal(train, _brute_force_train(groups, test, gap))


@pytest.mark.parametrize(
    "splitter, params, error, match",
    [
        (HVBlockCrossValidator, {"h": 1.5}, TypeError, "h must be an integer"),
        (HVBlockCrossValidator, {"v": -1}, ValueError, "v must be at least 0"),
        (
            BufferedBlockedKFold,
            {"n_splits": 1},
            ValueError,
            "n_splits must be at least",
        ),
        (BufferedBlockedKFold, {"gap": "1"}, TypeError, "gap must be an integer"),
    ],
)
def test_invalid_parameters(splitter, params, error, match):
    """Test that the splitter parameters must be large enough integers."""
    with pytest.raises(error, match=match):
        splitter(**params)


def test_invalid_data():
    """Test sequences too short for a test block and folds with no training data."""
    with pytest.raises(ValueError, match=r"at least 2 \* v \+ 1 cases"):
        HVBlockCrossValidator(v=1).get_n_splits(np.arange(5), groups=[0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="Reduce h or v"):
        next(HVBlockCrossValidator(h=2, v=1).split(np.arange(7)))
    # Other sequences keep training cases available.
    grouped = HVBlockCrossValidator(h=2).split(np.arange(4), groups=[0, 0, 1, 1])
    assert len(list(grouped)) == 4

    with pytest.raises(ValueError, match="smallest sequence size of 2"):
        next(
            BufferedBlockedKFold(n_splits=3).split(np.arange(5), groups=[0, 0, 1, 1, 1])
        )
    with pytest.raises(ValueError, match="Fold 1 has no training cases"):
        list(BufferedBlockedKFold(n_splits=3, gap=4).split(np.arange(10)))


@pytest.mark.parametrize(
    "cv", [HVBlockCrossValidator(h=1), BufferedBlockedKFold(n_splits=2)]
)
def test_sequence_ids(cv):
    """Test that missing IDs are rejected and IDs are not converted to strings."""
    X = np.arange(6)
    for groups in [
        np.array([0, 0, 0, np.nan, np.nan, np.nan]),
        ["a", "a", "a", np.nan, np.nan, np.nan],
        pd.Series([0, 0, 0, pd.NA, pd.NA, pd.NA], dtype="Int64"),
    ]:
        with pytest.raises(ValueError, match="must not contain missing values"):
            next(cv.split(X, groups=groups))

    expected = list(cv.split(X, groups=[0, 0, 0, 1, 1, 1]))
    _assert_splits_equal(list(cv.split(X, groups=[0, 0, 0, "0", "0", "0"])), expected)


@pytest.mark.parametrize(
    "cv", [HVBlockCrossValidator(h=1, v=1), BufferedBlockedKFold(n_splits=3, gap=1)]
)
def test_sklearn_metadata_routing(cv):
    """Test that sklearn routes groups to the splitters and scores their folds."""
    X = np.arange(12).reshape(-1, 1)
    # The jump at the sequence boundary makes grouped and ungrouped scores differ.
    y = np.array([0.0, 1, 2, 3, 4, 50, -40, 1, 2, 3, 4, 5])
    groups = np.repeat([0, 1], 6)
    expected = [
        -mean_squared_error(
            y[test], DummyRegressor().fit(X[train], y[train]).predict(X[test])
        )
        for train, test in cv.split(X, y, groups)
    ]
    ungrouped = cross_val_score(
        DummyRegressor(), X, y, cv=cv, scoring="neg_mean_squared_error"
    )
    assert not np.array_equal(ungrouped, expected)

    with config_context(enable_metadata_routing=True):
        scores = cross_val_score(
            DummyRegressor(),
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
            params={"groups": groups},
        )
        search = GridSearchCV(
            DummyRegressor(),
            {"strategy": ["mean", "median"]},
            cv=cv,
            scoring="neg_mean_squared_error",
        ).fit(X, y, groups=groups)

    np.testing.assert_allclose(scores, expected)
    np.testing.assert_allclose(
        search.cv_results_["mean_test_score"][0], np.mean(expected)
    )
