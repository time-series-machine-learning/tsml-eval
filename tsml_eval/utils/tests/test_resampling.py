"""Tests for dataset resampling functions."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]


import numpy as np
import pandas as pd
import pytest

from tsml_eval.datasets._loaders import (
    load_equal_minimal_japanese_vowels,
    load_minimal_chinatown,
    load_unequal_minimal_chinatown,
)
from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH
from tsml_eval.utils.resampling import (
    cv_fold_data,
    cv_fold_indices,
    cv_n_folds,
    cv_subjects_per_fold,
    resample_data,
    resample_data_indices,
    stratified_resample_data,
    stratified_resample_data_indices,
)
from tsml_eval.utils.results_validation import compare_result_file_resample


@pytest.mark.parametrize(
    "loader", [load_minimal_chinatown, load_equal_minimal_japanese_vowels]
)
def test_resample_data(loader):
    """Test resampling returns valid data."""
    X_train, y_train = loader(split="TRAIN")
    X_test, y_test = loader(split="TEST")

    train_size = X_train.shape
    test_size = X_test.shape

    X_train, y_train, X_test, y_test = resample_data(X_train, y_train, X_test, y_test)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert X_train.shape == train_size
    assert X_test.shape == test_size


def test_resample_data_unequal():
    """Test resampling returns valid data with unequal length input."""
    X_train, y_train = load_unequal_minimal_chinatown(split="TRAIN")
    X_test, y_test = load_unequal_minimal_chinatown(split="TEST")

    train_size = len(X_train)
    test_size = len(X_test)

    X_train, y_train, X_test, y_test = resample_data(X_train, y_train, X_test, y_test)

    assert isinstance(X_train, list)
    assert isinstance(X_train[0], np.ndarray)
    assert isinstance(X_test, list)
    assert isinstance(X_test[0], np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert len(X_train) == train_size
    assert len(X_test) == test_size


def test_resample_data_invalid():
    """Test resampling raises an error with invalid input."""
    X = pd.DataFrame(np.random.random((10, 10)))
    y = pd.Series(np.zeros(10))

    with pytest.raises(ValueError, match="X_train must be a"):
        resample_data(X, y, X, y)


def test_resample_data_indices():
    """Test resampling returns valid indices."""
    X_train, y_train = load_minimal_chinatown(split="TRAIN")
    X_test, y_test = load_minimal_chinatown(split="TEST")

    new_X_train, _, new_X_test, _ = resample_data(
        X_train, y_train, X_test, y_test, random_state=0
    )
    train_indices, test_indices = resample_data_indices(y_train, y_test, random_state=0)
    X = np.concatenate((X_train, X_test), axis=0)

    assert isinstance(train_indices, np.ndarray)
    assert isinstance(test_indices, np.ndarray)
    assert len(train_indices) == len(new_X_train)
    assert len(test_indices) == len(new_X_test)
    assert len(np.unique(np.concatenate((train_indices, test_indices), axis=0))) == len(
        X
    )
    assert (new_X_train[0] == X[train_indices[0]]).all()
    assert (new_X_test[0] == X[test_indices[0]]).all()


@pytest.mark.parametrize(
    "loader", [load_minimal_chinatown, load_equal_minimal_japanese_vowels]
)
def test_stratified_resample_data(loader):
    """Test stratified resampling returns valid data and class distribution."""
    X_train, y_train = loader(split="TRAIN")
    X_test, y_test = loader(split="TEST")

    train_size = X_train.shape
    test_size = X_test.shape
    _, counts_train = np.unique(y_train, return_counts=True)
    _, counts_test = np.unique(y_test, return_counts=True)

    X_train, y_train, X_test, y_test = stratified_resample_data(
        X_train, y_train, X_test, y_test
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert X_train.shape == train_size
    assert X_test.shape == test_size

    _, counts_train_new = np.unique(y_train, return_counts=True)
    _, counts_test_new = np.unique(y_test, return_counts=True)

    assert list(counts_train_new) == list(counts_train)
    assert list(counts_test_new) == list(counts_test)


def test_stratified_resample_data_unequal():
    """Test stratified resampling returns valid data with unequal length input."""
    X_train, y_train = load_unequal_minimal_chinatown(split="TRAIN")
    X_test, y_test = load_unequal_minimal_chinatown(split="TEST")

    train_size = len(X_train)
    test_size = len(X_test)
    _, counts_train = np.unique(y_train, return_counts=True)
    _, counts_test = np.unique(y_test, return_counts=True)

    X_train, y_train, X_test, y_test = stratified_resample_data(
        X_train, y_train, X_test, y_test
    )

    assert isinstance(X_train, list)
    assert isinstance(X_train[0], np.ndarray)
    assert isinstance(X_test, list)
    assert isinstance(X_test[0], np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert len(X_train) == train_size
    assert len(X_test) == test_size

    _, counts_train_new = np.unique(y_train, return_counts=True)
    _, counts_test_new = np.unique(y_test, return_counts=True)

    assert list(counts_train_new) == list(counts_train)
    assert list(counts_test_new) == list(counts_test)


def test_stratified_resample_data_invalid():
    """Test stratified resampling raises an error with invalid input."""
    X = pd.DataFrame(np.random.random((10, 10)))
    y = pd.Series(np.zeros(10))

    with pytest.raises(ValueError, match="X_train must be a"):
        stratified_resample_data(X, y, X, y)


def test_stratified_resample_data_indices():
    """Test stratified resampling returns valid indices."""
    X_train, y_train = load_minimal_chinatown(split="TRAIN")
    X_test, y_test = load_minimal_chinatown(split="TEST")

    new_X_train, _, new_X_test, _ = stratified_resample_data(
        X_train, y_train, X_test, y_test, random_state=0
    )
    train_indices, test_indices = stratified_resample_data_indices(
        y_train, y_test, random_state=0
    )
    X = np.concatenate((X_train, X_test), axis=0)

    assert isinstance(train_indices, np.ndarray)
    assert isinstance(test_indices, np.ndarray)
    assert len(train_indices) == len(new_X_train)
    assert len(test_indices) == len(new_X_test)
    assert len(np.unique(np.concatenate((train_indices, test_indices), axis=0))) == len(
        X
    )
    assert (new_X_train[0] == X[train_indices[0]]).all()
    assert (new_X_test[0] == X[test_indices[0]]).all()


@pytest.mark.parametrize(
    "paths",
    [
        [
            _TEST_RESULTS_PATH + "/classification/classificationResultsFile1.csv",
            _TEST_RESULTS_PATH + "/classification/classificationResultsFile1.csv",
            True,
        ],
        [
            _TEST_RESULTS_PATH + "/classification/classificationResultsFile1.csv",
            _TEST_RESULTS_PATH + "/classification/classificationResultsFile2.csv",
            False,
        ],
    ],
)
def test_compare_result_file_resample(paths):
    """Test compare result file resample function."""
    assert compare_result_file_resample(paths[0], paths[1]) == paths[2]


def test_compare_result_file_resample_invalid():
    """Test compare result file resample function with invalid input."""
    p1 = _TEST_RESULTS_PATH + "/classification/classificationResultsFile1.csv"
    p3 = _TEST_RESULTS_PATH + "/classification/classificationResultsFile3.csv"

    with pytest.raises(ValueError, match="Input results file have different"):
        compare_result_file_resample(p1, p3)


def _cv_labels():
    rng = np.random.RandomState(1)
    y = rng.choice(["a", "b", "c"], 120)
    return y[:70], y[70:]


def test_cv_folds_partition_cases():
    """Each case is in exactly one test fold, and folds are stratified."""
    y_train, y_test = _cv_labels()
    seen = []
    for fold in range(10):
        train, test = cv_fold_indices(y_train, y_test, fold)
        assert len(set(train) & set(test)) == 0
        assert len(train) + len(test) == 120
        seen.extend(test)
    assert sorted(seen) == list(range(120))
    a, b = cv_fold_indices(y_train, y_test, 3)
    c, d = cv_fold_indices(y_train, y_test, 3)
    assert np.array_equal(a, c) and np.array_equal(b, d)


@pytest.mark.parametrize(
    "n_subjects,per_fold,expected",
    [(9, 1, 9), (14, 1, 14), (15, 1, 15), (16, 2, 8), (29, 3, 10), (32, 3, 11)],
)
def test_cv_folds_by_subject(n_subjects, per_fold, expected):
    """Subjects never span train and test; equal subjects per fold but the last."""
    y_train, y_test = _cv_labels()
    groups = np.arange(120) % n_subjects
    assert cv_subjects_per_fold(n_subjects) == per_fold
    assert cv_n_folds(10, groups) == expected
    seen, sizes = [], []
    for fold in range(expected):
        train, test = cv_fold_indices(y_train, y_test, fold, groups=groups)
        assert not set(groups[train]) & set(groups[test])
        seen.extend(test)
        sizes.append(len(set(groups[test])))
    assert sorted(seen) == list(range(120))
    assert sizes[:-1] == [per_fold] * (expected - 1)
    assert 1 <= sizes[-1] <= per_fold
    with pytest.raises(ValueError, match="fold must be"):
        cv_fold_indices(y_train, y_test, expected, groups=groups)


def test_cv_folds_bad_groups():
    """Group IDs must match the cases."""
    y_train, y_test = _cv_labels()
    with pytest.raises(ValueError, match="one per case"):
        cv_fold_indices(y_train, y_test, 0, groups=np.zeros(5))


@pytest.mark.parametrize(
    "loader", [load_minimal_chinatown, load_unequal_minimal_chinatown]
)
def test_cv_fold_data(loader):
    """Fold data matches the fold indices, for arrays and lists."""
    X_train, y_train = loader(split="TRAIN")
    X_test, y_test = loader(split="TEST")
    groups = np.arange(len(y_train) + len(y_test)) // 3
    train_X, train_y, test_X, test_y = cv_fold_data(
        X_train, y_train, X_test, y_test, 2, groups=groups
    )
    train, test = cv_fold_indices(y_train, y_test, 2, groups=groups)
    y = np.concatenate((y_train, y_test))
    assert np.array_equal(train_y, y[train]) and np.array_equal(test_y, y[test])
    assert len(train_X) == len(train) and len(test_X) == len(test)
