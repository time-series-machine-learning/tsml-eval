"""Tests for dependent time-series cross-validation utilities."""

import enum

import numpy as np
import pandas as pd
import pytest
from aeon.classification import DummyClassifier as AeonDummyClassifier
from aeon.regression import DummyRegressor as AeonDummyRegressor
from sklearn import config_context
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)

from tsml_eval.experiments.cross_validation import (
    classification_cross_validation,
    classification_cross_validation_folds,
    regression_cross_validation,
    regression_cross_validation_folds,
)
from tsml_eval.utils.cross_validation import (
    BufferedBlockedKFold,
    HVBlockCrossValidator,
)
from tsml_eval.utils.tests.test_results_writing import (
    _check_classification_file_format,
    _check_regression_file_format,
)


class _SequenceID(enum.Enum):
    """Enum sequence identifier for hashable non-orderable group testing."""

    FIRST = 1
    SECOND = 2


def _assert_split_equal(split, expected_train, expected_test):
    """Assert that a generated split contains the expected indices."""
    train, test = split
    np.testing.assert_array_equal(train, expected_train)
    np.testing.assert_array_equal(test, expected_test)


@pytest.mark.parametrize(
    "cv", [HVBlockCrossValidator(h=1), BufferedBlockedKFold(n_splits=2)]
)
@pytest.mark.parametrize(
    "groups",
    [
        np.array([0, 0, 0, np.nan, np.nan, np.nan]),
        np.array([0, 0, 0, np.nan, np.nan, np.nan], dtype=object),
        ["a", "a", "a", np.nan, np.nan, np.nan],
        ["a", "a", "a", None, None, None],
        ["a", "a", "a", pd.NA, pd.NA, pd.NA],
        np.array(["2020-01-01"] * 3 + ["NaT"] * 3, dtype="datetime64[D]"),
        pd.Series([0, 0, 0, pd.NA, pd.NA, pd.NA], dtype="Int64"),
        pd.Series(["a", "a", "a", pd.NA, pd.NA, pd.NA], dtype="string"),
        pd.Categorical(["a", "a", "a", None, None, None]),
    ],
    ids=[
        "float",
        "object",
        "string-list-nan",
        "none",
        "list-pd-na",
        "nat",
        "nullable-int",
        "nullable-string",
        "categorical",
    ],
)
def test_missing_sequence_ids_are_rejected(cv, groups):
    """Reject missing IDs before they can create spurious independent sequences."""
    X = np.arange(6)
    with pytest.raises(ValueError, match="groups must not contain missing values"):
        next(cv.split(X, groups=groups))
    if isinstance(cv, HVBlockCrossValidator):
        with pytest.raises(ValueError, match="groups must not contain missing values"):
            cv.get_n_splits(X, groups=groups)


@pytest.mark.parametrize(
    "cv", [HVBlockCrossValidator(h=1), BufferedBlockedKFold(n_splits=2)]
)
@pytest.mark.parametrize(
    "groups",
    [
        ["a", "a", "a", "nan", "nan", "nan"],
        [0, 0, 0, "0", "0", "0"],
        pd.Series([101, 101, 101, "102A", "102A", "102A"]),
        [_SequenceID.FIRST] * 3 + [_SequenceID.SECOND] * 3,
        pd.Series([2, 2, 2, 1, 1, 1], dtype="Int64"),
        pd.Series(["b", "b", "b", "a", "a", "a"], dtype="string"),
        pd.Categorical(
            ["b", "b", "b", "a", "a", "a"],
            categories=["unused", "a", "b"],
        ),
    ],
    ids=[
        "literal-nan-string",
        "distinct-int-and-str-zero",
        "mixed-int-str-series",
        "enum-ids",
        "nullable-int",
        "nullable-string",
        "categorical-unused-category",
    ],
)
def test_valid_sequence_id_types(cv, groups):
    """Support valid string, nullable, categorical, enum, and mixed sequence IDs."""
    X = np.arange(6)
    actual = list(cv.split(X, groups=groups))
    expected = list(cv.split(X, groups=[0, 0, 0, 1, 1, 1]))
    assert len(actual) == len(expected)
    if isinstance(cv, HVBlockCrossValidator):
        assert cv.get_n_splits(X, groups=groups) == len(expected)
    for split, (train, test) in zip(actual, expected):
        _assert_split_equal(split, train, test)


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


@pytest.mark.parametrize("h", [0, 1, 2])
@pytest.mark.parametrize("v", [0, 1, 2])
def test_hv_block_exact_buffer_invariants(h, v):
    """Verify exact within-sequence buffer exclusion and cross-sequence retention."""
    groups = np.array(["seq_b", "seq_a"] * 5 + ["seq_b", "seq_b"])
    X = np.arange(len(groups))
    cv = HVBlockCrossValidator(h=h, v=v)

    splits = list(cv.split(X, groups=groups))
    assert len(splits) == cv.get_n_splits(X, groups=groups)
    assert [int(test[0]) for _, test in splits] == sorted(
        int(test[0]) for _, test in splits
    )

    seq_positions = {
        seq_id: np.flatnonzero(groups == seq_id) for seq_id in ["seq_a", "seq_b"]
    }
    for train, test in splits:
        assert len(train) >= 1
        assert len(test) == 2 * v + 1
        assert np.all(np.diff(train) > 0)
        assert np.all(np.diff(test) > 0)
        assert np.intersect1d(train, test).size == 0

        test_seq_ids = set(groups[test])
        assert len(test_seq_ids) == 1
        active_seq = next(iter(test_seq_ids))
        seq_idx = seq_positions[active_seq]

        test_pos = np.searchsorted(seq_idx, test)
        np.testing.assert_array_equal(
            test_pos, np.arange(test_pos[0], test_pos[0] + 2 * v + 1)
        )

        expected_active_train = [
            seq_idx[pos]
            for pos in range(len(seq_idx))
            if np.min(np.abs(test_pos - pos)) > h
        ]
        other_seq = "seq_b" if active_seq == "seq_a" else "seq_a"
        expected_train = np.sort(
            np.concatenate([expected_active_train, seq_positions[other_seq]])
        ).astype(int)
        np.testing.assert_array_equal(train, expected_train)


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
    """Test invalid hv-block data and parity between get_n_splits and split."""
    cv = HVBlockCrossValidator(v=1)
    with pytest.raises(TypeError, match="required positional argument: 'X'"):
        cv.get_n_splits()
    for n_cases in [0, 1]:
        with pytest.raises(ValueError, match="at least 2 cases"):
            HVBlockCrossValidator().get_n_splits(np.arange(n_cases))
        with pytest.raises(ValueError, match="at least 2 cases"):
            next(HVBlockCrossValidator().split(np.arange(n_cases)))
    with pytest.raises(ValueError, match=r"at least 2 \* v \+ 1 cases"):
        cv.get_n_splits(np.arange(2))
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        next(cv.split(np.arange(4), groups=[0, 0, 0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        next(cv.split(np.arange(4), groups=np.zeros((4, 1))))
    with pytest.raises(ValueError, match="one-dimensional"):
        next(cv.split(np.arange(4), groups=[[0], [0], [1], [1]]))
    with pytest.raises(ValueError, match=r"at least 2 \* v \+ 1 cases"):
        cv.get_n_splits(np.arange(5), groups=[0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match=r"at least 2 \* v \+ 1 cases"):
        next(cv.split(np.arange(5), groups=[0, 0, 1, 1, 1]))
    with pytest.raises(ValueError, match="Reduce h or v"):
        HVBlockCrossValidator(h=2, v=1).get_n_splits(np.arange(7))
    with pytest.raises(ValueError, match="Reduce h or v"):
        next(HVBlockCrossValidator(h=2, v=1).split(np.arange(7)))
    with pytest.raises(ValueError, match="Reduce h"):
        HVBlockCrossValidator(h=2, v=0).get_n_splits(np.arange(5))
    with pytest.raises(ValueError, match="Reduce h"):
        next(HVBlockCrossValidator(h=2, v=0).split(np.arange(5)))

    # Multiple independent sequences keep training cases available across sequences.
    grouped_cv = HVBlockCrossValidator(h=2, v=0)
    groups = [0, 0, 1, 1]
    assert grouped_cv.get_n_splits(np.arange(4), groups=groups) == 4
    assert len(list(grouped_cv.split(np.arange(4), groups=groups))) == 4


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


@pytest.mark.parametrize("n_splits", [2, 3, 4])
@pytest.mark.parametrize("gap", [0, 1, 2])
def test_buffered_blocked_kfold_exact_buffer_invariants(n_splits, gap):
    """Verify partition coverage and exact per-sequence gap exclusion."""
    groups = np.array(["s2", "s1", "s3"] * 6 + ["s1", "s2", "s1"])
    X = np.arange(len(groups))
    cv = BufferedBlockedKFold(n_splits=n_splits, gap=gap)

    splits = list(cv.split(X, groups=groups))
    assert len(splits) == n_splits

    all_test = np.concatenate([test for _, test in splits])
    np.testing.assert_array_equal(np.sort(all_test), np.arange(len(X)))

    seq_positions = {
        seq_id: np.flatnonzero(groups == seq_id) for seq_id in ["s1", "s2", "s3"]
    }
    for train, test in splits:
        assert len(train) >= 1
        assert np.all(np.diff(train) > 0)
        assert np.all(np.diff(test) > 0)
        assert np.intersect1d(train, test).size == 0

        expected_train_parts = []
        for seq_idx in seq_positions.values():
            seq_test = np.intersect1d(test, seq_idx)
            assert len(seq_test) >= 1
            test_pos = np.searchsorted(seq_idx, seq_test)
            np.testing.assert_array_equal(
                test_pos, np.arange(test_pos[0], test_pos[-1] + 1)
            )
            expected_train_parts.append(
                [
                    seq_idx[pos]
                    for pos in range(len(seq_idx))
                    if np.min(np.abs(test_pos - pos)) > gap
                ]
            )
        expected_train = np.sort(np.concatenate(expected_train_parts)).astype(int)
        np.testing.assert_array_equal(train, expected_train)


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
        HVBlockCrossValidator(h=1, v=0),
        HVBlockCrossValidator(h=1, v=1),
        BufferedBlockedKFold(n_splits=3, gap=1),
    ],
)
def test_yielded_fold_arrays_do_not_alias(cv):
    """Ensure mutating yielded train/test arrays never corrupts other folds."""
    X = np.arange(10)
    groups = np.repeat([0, 1], 5)
    reference = [
        (train.copy(), test.copy()) for train, test in cv.split(X, groups=groups)
    ]

    # Mutate in place during lazy iteration.
    observed_during_iteration = []
    for train, test in cv.split(X, groups=groups):
        assert not np.shares_memory(train, test)
        observed_during_iteration.append((train.copy(), test.copy()))
        train[:] = -1
        test[:] = -1

    for actual, (exp_train, exp_test) in zip(observed_during_iteration, reference):
        _assert_split_equal(actual, exp_train, exp_test)

    # Mutate after collecting all splits (overlapping test windows when v > 0).
    materialised = list(cv.split(X, groups=groups))
    for i, (train_i, test_i) in enumerate(materialised):
        for train_j, test_j in materialised[i + 1 :]:
            assert not np.shares_memory(train_i, train_j)
            assert not np.shares_memory(test_i, test_j)
            assert not np.shares_memory(train_i, test_j)
    materialised[0][0][:] = -99
    materialised[0][1][:] = -99
    for actual, (exp_train, exp_test) in zip(materialised[1:], reference[1:]):
        _assert_split_equal(actual, exp_train, exp_test)


@pytest.mark.parametrize(
    "cv",
    [
        HVBlockCrossValidator(h=1, v=1),
        BufferedBlockedKFold(n_splits=3, gap=1),
    ],
)
def test_time_series_containers(cv):
    """Support 3D arrays, unequal-length series lists, and indexed DataFrames."""
    groups = np.repeat([0, 1], 6)
    expected = list(cv.split(np.arange(12), groups=groups))

    containers = [
        np.zeros((12, 2, 15)),
        [np.zeros((2, 5 + i)) for i in range(12)],
        pd.DataFrame({"a": np.arange(12)}, index=np.arange(100, 112)),
    ]
    indexed_groups = pd.Series(groups, index=np.arange(200, 212))

    for X in containers:
        splits = list(cv.split(X, groups=indexed_groups))
        assert len(splits) == len(expected)
        for split, (exp_train, exp_test) in zip(splits, expected):
            _assert_split_equal(split, exp_train, exp_test)


@pytest.mark.parametrize(
    "cv",
    [
        HVBlockCrossValidator(h=1, v=1),
        BufferedBlockedKFold(n_splits=3, gap=1),
    ],
)
def test_sklearn_cv_execution_and_metadata_routing(cv):
    """Verify sklearn CV runners execute the exact grouped folds and scores."""
    X = np.arange(12).reshape(-1, 1)
    # Piecewise target with a jump at the sequence boundary so grouped vs ungrouped
    # buffering changes the training set around cases 4..7 and alters fold MSEs.
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 50.0, -40.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    groups = np.repeat([0, 1], 6)

    expected_folds = list(cv.split(X, y, groups))
    manual_scores = np.array(
        [
            -mean_squared_error(
                y[test], DummyRegressor().fit(X[train], y[train]).predict(X[test])
            )
            for train, test in expected_folds
        ]
    )
    ungrouped_scores = cross_val_score(
        DummyRegressor(), X, y, cv=cv, scoring="neg_mean_squared_error"
    )

    with config_context(enable_metadata_routing=True):
        routed_scores = cross_val_score(
            DummyRegressor(),
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
            params={"groups": groups},
        )
        cv_dict = cross_validate(
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

    assert len(routed_scores) == cv.get_n_splits(X, y, groups)
    np.testing.assert_allclose(routed_scores, manual_scores)
    np.testing.assert_allclose(cv_dict["test_score"], manual_scores)
    assert not np.array_equal(routed_scores, ungrouped_scores)
    np.testing.assert_allclose(
        search.cv_results_["mean_test_score"][0], manual_scores.mean()
    )


def test_buffered_blocked_kfold_cross_val_predict():
    """Verify out-of-fold predictions from cross_val_predict match manual folds."""
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10, dtype=float) ** 2
    groups = np.repeat(["seq_a", "seq_b"], 5)
    cv = BufferedBlockedKFold(n_splits=2, gap=1)

    preds = cross_val_predict(DummyRegressor(), X, y, cv=cv, groups=groups)
    expected = np.empty_like(y)
    for train, test in cv.split(X, y, groups):
        expected[test] = DummyRegressor().fit(X[train], y[train]).predict(X[test])

    np.testing.assert_allclose(preds, expected)


@pytest.mark.parametrize(
    "cv",
    [
        HVBlockCrossValidator(h=1, v=1),
        BufferedBlockedKFold(n_splits=3, gap=1),
    ],
)
def test_tsml_eval_cross_validation_runners(tmp_path, cv):
    """Verify tsml-eval classification and regression CV runners with splitters."""
    X = np.arange(36, dtype=float).reshape(12, 1, 3)
    y_clf = np.tile([0, 1], 6)
    y_reg = np.linspace(0.0, 5.0, 12)
    expected_folds = list(cv.split(X))

    clf_folds = classification_cross_validation_folds(X, y_clf, cv=cv)
    reg_folds = regression_cross_validation_folds(X, y_reg, cv=cv)
    assert len(clf_folds) == len(expected_folds)
    assert len(reg_folds) == len(expected_folds)
    for actual_clf, actual_reg, (exp_train, exp_test) in zip(
        clf_folds, reg_folds, expected_folds
    ):
        _assert_split_equal(actual_clf, exp_train, exp_test)
        _assert_split_equal(actual_reg, exp_train, exp_test)

    clf_dir = tmp_path / "clf"
    classification_cross_validation(
        X,
        y_clf,
        AeonDummyClassifier(),
        str(clf_dir),
        cv=cv,
        classifier_name="DummyHVClf",
        dataset_name="SyntheticClf",
        benchmark_time=False,
    )
    for fold_idx, (_, test) in enumerate(expected_folds):
        test_file = (
            clf_dir
            / "DummyHVClf"
            / "Predictions"
            / "SyntheticClf"
            / f"testResample{fold_idx}.csv"
        )
        assert test_file.exists()
        _check_classification_file_format(str(test_file), num_results_lines=len(test))

    reg_dir = tmp_path / "reg"
    regression_cross_validation(
        X,
        y_reg,
        AeonDummyRegressor(),
        str(reg_dir),
        cv=cv,
        regressor_name="DummyHVReg",
        dataset_name="SyntheticReg",
        benchmark_time=False,
    )
    for fold_idx, (_, test) in enumerate(expected_folds):
        test_file = (
            reg_dir
            / "DummyHVReg"
            / "Predictions"
            / "SyntheticReg"
            / f"testResample{fold_idx}.csv"
        )
        assert test_file.exists()
        _check_regression_file_format(str(test_file), num_results_lines=len(test))


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
    with pytest.raises(ValueError, match="one-dimensional"):
        next(cv.split(np.arange(6), groups=[[0], [0], [0], [1], [1], [1]]))
    with pytest.raises(ValueError, match="at least one case"):
        next(cv.split(np.arange(0)))
    with pytest.raises(ValueError, match="increase n_splits"):
        next(BufferedBlockedKFold(n_splits=3, gap=4).split(np.arange(10)))
