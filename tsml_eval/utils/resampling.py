"""Utility functions fordata resampling."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = [
    "resample_data",
    "resample_data_indices",
    "stratified_resample_data",
    "stratified_resample_data_indices",
    "cv_subjects_per_fold",
    "cv_n_folds",
    "cv_fold_indices",
    "cv_fold_data",
]


import warnings

import numpy as np
from sklearn.utils import check_random_state


def resample_data(X_train, y_train, X_test, y_test, random_state=None):
    """Resample data without replacement using a random state.

    Reproducible resampling. Combines train and test, randomly resamples, then returns
    new train and test.

    Parameters
    ----------
    X_train : np.ndarray or list of np.ndarray
        Train data in a 2d or 3d ndarray or list of arrays.
    y_train : np.ndarray
        Train data labels.
    X_test : np.ndarray or list of np.ndarray
        Test data in a 2d or 3d ndarray or list of arrays.
    y_test : np.ndarray
        Test data labels.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    train_X : np.ndarray or list of np.ndarray
        New train data.
    train_y : np.ndarray
        New train labels.
    test_X : np.ndarray or list of np.ndarray
        New test data.
    test_y : np.ndarray
        New test labels.
    """
    if isinstance(X_train, np.ndarray):
        is_array = True
    elif isinstance(X_train, list):
        is_array = False
    else:
        raise ValueError(
            "X_train must be a np.ndarray array or list of np.ndarray arrays"
        )

    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = (
        np.concatenate([X_train, X_test], axis=0) if is_array else X_train + X_test
    )

    # shuffle data indices
    rng = check_random_state(random_state)
    indices = np.arange(len(all_data), dtype=int)
    rng.shuffle(indices)

    train_indices = indices[: len(X_train)]
    test_indices = indices[len(X_train) :]

    # split the shuffled data into train and test
    X_train = (
        all_data[train_indices] if is_array else [all_data[i] for i in train_indices]
    )
    y_train = all_labels[train_indices]
    X_test = all_data[test_indices] if is_array else [all_data[i] for i in test_indices]
    y_test = all_labels[test_indices]

    return X_train, y_train, X_test, y_test


def resample_data_indices(y_train, y_test, random_state=None):
    """Return data resample indices without replacement using a random state.

    Reproducible resampling. Combines train and test, randomly resamples, then returns
    the new position for both the train and test set. Uses indices for a combined train
    and test set, with test indices appearing after train indices.

    Parameters
    ----------
    y_train : np.ndarray
        Train data labels.
    y_test : np.ndarray
        Test data labels.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    train_indices : np.ndarray
        The index of cases to use in the train set from the combined train and test
        data.
    test_indices : np.ndarray
        The index of cases to use in the test set from the combined train and test data.
    """
    # shuffle data indices
    rng = check_random_state(random_state)
    indices = np.arange(len(y_train) + len(y_test), dtype=int)
    rng.shuffle(indices)

    train_indices = indices[: len(y_train)]
    test_indices = indices[len(y_train) :]

    return train_indices, test_indices


def stratified_resample_data(X_train, y_train, X_test, y_test, random_state=None):
    """Stratified resample data without replacement using a random state.

    Reproducible resampling. Combines train and test, resamples to get the same class
    distribution, then returns new train and test.

    Parameters
    ----------
    X_train : np.ndarray or list of np.ndarray
        Train data in a 2d or 3d ndarray or list of arrays.
    y_train : np.ndarray
        Train data labels.
    X_test : np.ndarray or list of np.ndarray
        Test data in a 2d or 3d ndarray or list of arrays.
    y_test : np.ndarray
        Test data labels.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    train_X : np.ndarray or list of np.ndarray
        New train data.
    train_y : np.ndarray
        New train labels.
    test_X : np.ndarray or list of np.ndarray
        New test data.
    test_y : np.ndarray
        New test labels.
    """
    if isinstance(X_train, np.ndarray):
        is_array = True
    elif isinstance(X_train, list):
        is_array = False
    else:
        raise ValueError(
            "X_train must be a np.ndarray array or list of np.ndarray arrays"
        )

    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = (
        np.concatenate([X_train, X_test], axis=0) if is_array else X_train + X_test
    )

    # shuffle data indices
    rng = check_random_state(random_state)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    # ensure same classes exist in both train and test
    assert list(unique_train) == list(unique_test)

    if is_array:
        shape = list(X_train.shape)
        shape[0] = 0

    X_train = np.zeros(shape) if is_array else []
    y_train = np.zeros(0)
    X_test = np.zeros(shape) if is_array else []
    y_test = np.zeros(0)

    # for each class
    for label_index in range(len(unique_train)):
        # get the indices of all instances with this class label and shuffle them
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]
        rng.shuffle(indices)

        train_indices = indices[: counts_train[label_index]]
        test_indices = indices[counts_train[label_index] :]

        # extract data from corresponding indices
        train_cases = (
            all_data[train_indices]
            if is_array
            else [all_data[i] for i in train_indices]
        )
        train_labels = all_labels[train_indices]
        test_cases = (
            all_data[test_indices] if is_array else [all_data[i] for i in test_indices]
        )
        test_labels = all_labels[test_indices]

        # concat onto current data from previous loop iterations
        X_train = (
            np.concatenate([X_train, train_cases], axis=0)
            if is_array
            else X_train + train_cases
        )
        y_train = np.concatenate([y_train, train_labels], axis=None)
        X_test = (
            np.concatenate([X_test, test_cases], axis=0)
            if is_array
            else X_test + test_cases
        )
        y_test = np.concatenate([y_test, test_labels], axis=None)

    return X_train, y_train, X_test, y_test


def stratified_resample_data_indices(y_train, y_test, random_state=None):
    """Return stratified data resample indices without replacement using a random state.

    Reproducible resampling. Combines train and test, resamples to get the same class
    distribution, then returns the new position for both the train and test set.
    Uses indices for a combined train and test set, with test indices appearing after
    train indices.

    Parameters
    ----------
    y_train : np.ndarray
        Train data labels.
    y_test : np.ndarray
        Test data labels.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    train_indices : np.ndarray
        The index of cases to use in the train set from the combined train and test
        data.
    test_indices : np.ndarray
        The index of cases to use in the test set from the combined train and test data.
    """
    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)

    # shuffle data indices
    rng = check_random_state(random_state)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    # ensure same classes exist in both train and test
    assert list(unique_train) == list(unique_test)

    train_indices = np.zeros(0, dtype=int)
    test_indices = np.zeros(0, dtype=int)

    # for each class
    for label_index in range(len(unique_train)):
        # get the indices of all instances with this class label and shuffle them
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]
        rng.shuffle(indices)

        train_indices = np.concatenate(
            [train_indices, indices[: counts_train[label_index]]], axis=None
        )
        test_indices = np.concatenate(
            [test_indices, indices[counts_train[label_index] :]], axis=None
        )

    return train_indices, test_indices


def cv_subjects_per_fold(n_subjects, n_folds=10):
    """Return how many subjects each cross-validation test fold holds.

    Up to ``n_folds + 5`` subjects this is 1 (leave one subject out). Above that it
    is ``n_subjects / n_folds`` rounded, so there are close to ``n_folds`` folds.
    Every fold holds this many subjects except the last, which holds the rest.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_folds : int, default=10
        Number of folds aimed for.

    Returns
    -------
    subjects_per_fold : int
        Subjects in each test fold but the last.
    """
    if n_subjects <= n_folds + 5:
        return 1
    return int(n_subjects / n_folds + 0.5)


def cv_n_folds(n_folds, groups=None):
    """Return the number of cross-validation folds actually used.

    Without groups this is ``n_folds``. With groups (subject IDs) every subject is
    held out whole, ``cv_subjects_per_fold`` subjects to a fold, so the number of
    folds is the number of subjects divided by that, rounded up.

    Parameters
    ----------
    n_folds : int
        Number of folds aimed for.
    groups : np.ndarray or None, default=None
        Group (subject) ID of each case.

    Returns
    -------
    n_folds : int
        Number of folds.
    """
    if groups is None:
        return n_folds
    n_subjects = len(np.unique(groups))
    per_fold = cv_subjects_per_fold(n_subjects, n_folds)
    return -(-n_subjects // per_fold)


def cv_fold_indices(y_train, y_test, fold, n_folds=10, groups=None, random_state=0):
    """Return train and test indices of one stratified cross-validation fold.

    Combines train and test, then returns fold ``fold`` (from 0) as the test set.
    Without ``groups`` the folds are ``StratifiedKFold(n_folds)``. With ``groups``
    (subject IDs) every subject is kept on one side of the split: subjects are put
    in a seeded random order, spread by their most common class so folds are close
    to stratified, then taken ``cv_subjects_per_fold`` at a time, the last fold
    holding the rest. There are ``cv_n_folds(n_folds, groups)`` folds. The split
    depends only on the labels, groups and ``random_state``, so every estimator gets
    the same folds.

    Parameters
    ----------
    y_train : np.ndarray
        Train data labels.
    y_test : np.ndarray
        Test data labels.
    fold : int
        The fold to return, from 0.
    n_folds : int, default=10
        Number of folds aimed for.
    groups : np.ndarray or None, default=None
        Group (subject) ID of each case in the combined train and test data, train
        cases first.
    random_state : int, default=0
        Seed for assigning cases (or groups) to folds.

    Returns
    -------
    train_indices : np.ndarray
        The index of cases to use in the train set from the combined train and test
        data.
    test_indices : np.ndarray
        The index of cases to use in the test set from the combined train and test data.
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.concatenate((y_train, y_test), axis=None)
    k = cv_n_folds(n_folds, groups)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError(
                f"{len(groups)} group IDs for {len(y)} cases; there must be one "
                "per case"
            )
        if k < 2:
            raise ValueError("Cross-validation by group needs at least two groups")
    if not 0 <= fold < k:
        raise ValueError(f"fold must be between 0 and {k - 1}, not {fold}")

    if groups is None:
        splitter = StratifiedKFold(k, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            # a class with fewer cases than folds is fine, it is absent from some
            warnings.filterwarnings("ignore", message="The least populated class")
            splits = list(splitter.split(np.zeros(len(y)), y))
        return splits[fold]

    rng = check_random_state(random_state)
    subjects = np.unique(groups)
    per_fold = cv_subjects_per_fold(len(subjects), n_folds)
    # most common class of each subject
    labels = []
    for subject in subjects:
        values, counts = np.unique(y[groups == subject], return_counts=True)
        labels.append(values[np.argmax(counts)])
    labels = np.array(labels)
    # shuffle within each class, then interleave the classes evenly
    position = np.zeros(len(subjects))
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        position[rng.permutation(members)] = (np.arange(len(members)) + 0.5) / len(
            members
        )
    order = subjects[np.lexsort((rng.random_sample(len(subjects)), position))]
    test = np.isin(groups, order[fold * per_fold : (fold + 1) * per_fold])
    return np.flatnonzero(~test), np.flatnonzero(test)


def cv_fold_data(
    X_train, y_train, X_test, y_test, fold, n_folds=10, groups=None, random_state=0
):
    """Return one stratified cross-validation fold of the combined data.

    See ``cv_fold_indices`` for how folds are made.

    Parameters
    ----------
    X_train : np.ndarray or list of np.ndarray
        Train data in a 2d or 3d ndarray or list of arrays.
    y_train : np.ndarray
        Train data labels.
    X_test : np.ndarray or list of np.ndarray
        Test data in a 2d or 3d ndarray or list of arrays.
    y_test : np.ndarray
        Test data labels.
    fold : int
        The fold to use as the test set, from 0.
    n_folds : int, default=10
        Number of folds asked for.
    groups : np.ndarray or None, default=None
        Group (subject) ID of each case, train cases first.
    random_state : int, default=0
        Seed for assigning cases (or groups) to folds.

    Returns
    -------
    train_X : np.ndarray or list of np.ndarray
        New train data.
    train_y : np.ndarray
        New train labels.
    test_X : np.ndarray or list of np.ndarray
        New test data.
    test_y : np.ndarray
        New test labels.
    """
    train_idx, test_idx = cv_fold_indices(
        y_train, y_test, fold, n_folds, groups, random_state
    )
    y = np.concatenate((y_train, y_test), axis=None)
    if isinstance(X_train, np.ndarray) and isinstance(X_test, np.ndarray):
        X = np.concatenate((X_train, X_test), axis=0)
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    X = list(X_train) + list(X_test)
    return (
        [X[i] for i in train_idx],
        y[train_idx],
        [X[i] for i in test_idx],
        y[test_idx],
    )
