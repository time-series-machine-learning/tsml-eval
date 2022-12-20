# -*- coding: utf-8 -*-

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os

import numpy as np
import pandas as pd
import sklearn


def resample(train_X, train_y, test_X, test_y, random_state):
    """Resample data without replacement using a random state.

    Reproducable resampling. Combines train and test, resamples to get the same class
    distribution, then returns new train and test.

    Parameters
    ----------
    train_X : pd.DataFrame
        train data attributes in sktime pandas format.
    train_y : np.array
        train data class labels.
    test_X : pd.DataFrame
        test data attributes in sktime pandas format.
    test_y : np.array
        test data class labels as np array.
    random_state : int
        seed to enable reproducable resamples

    Returns
    -------
    new train and test attributes and class labels.
    """
    all_targets = np.concatenate((train_y, test_y), axis=None)
    all_data = pd.concat([train_X, test_X])

    # add the target labeleds to the dataset
    all_data["target"] = all_targets

    # randomly shuffle all instances
    shuffled = all_data.sample(frac=1, random_state=random_state)

    # extract and remove the target column
    all_targets = shuffled["target"].to_numpy()
    shuffled = shuffled.drop("target", axis=1)

    # split the shuffled data into train and test
    train_cases = train_y.size
    train_X = shuffled.iloc[:train_cases]
    test_X = shuffled.iloc[train_cases:]
    train_y = all_targets[:train_cases]
    test_y = all_targets[train_cases:]

    # reset indices and return
    train_X = train_X.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)
    return train_X, train_y, test_X, test_y


def stratified_resample(X_train, y_train, X_test, y_test, random_state):
    """Stratified resample data without replacement using a random state.

    Reproducable resampling. Combines train and test, resamples to get the same class
    distribution, then returns new train and test.

    Parameters
    ----------
    X_train : pd.DataFrame
        train data attributes in sktime pandas format.
    y_train : np.array
        train data class labels.
    X_test : pd.DataFrame
        test data attributes in sktime pandas format.
    y_test : np.array
        test data class labels as np array.
    random_state : int
        seed to enable reproducable resamples
    Returns
    -------
    new train and test attributes and class labels.
    """
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = pd.concat([X_train, X_test])

    random_state = sklearn.utils.check_random_state(random_state)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    # ensure same classes exist in both train and test
    assert list(unique_train) == list(unique_test)

    X_train = pd.DataFrame()
    y_train = np.array([])
    X_test = pd.DataFrame()
    y_test = np.array([])

    # for each class
    for label_index in range(0, len(unique_train)):
        # get the indices of all instances with this class label
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]

        # shuffle them
        random_state.shuffle(indices)

        # take the first lot of instances for train, remainder for test
        num_instances = counts_train[label_index]
        train_indices = indices[0:num_instances]
        test_indices = indices[num_instances:]

        # extract data from corresponding indices
        train_instances = all_data.iloc[train_indices, :]
        test_instances = all_data.iloc[test_indices, :]
        train_labels = all_labels[train_indices]
        test_labels = all_labels[test_indices]

        # concat onto current data from previous loop iterations
        X_train = pd.concat([X_train, train_instances])
        X_test = pd.concat([X_test, test_instances])
        y_train = np.concatenate([y_train, train_labels], axis=None)
        y_test = np.concatenate([y_test, test_labels], axis=None)

    # reset indices and return
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    return X_train, y_train, X_test, y_test


def results_present(path, estimator, dataset, res):
    full_path = f"{path}/{estimator}/Predictions/{dataset}/testResample{res}.csv"
    full_path2 = f"{path}/{estimator}/Predictions/{dataset}/trainResample{res}.csv"
    if os.path.exists(full_path) and os.path.exists(full_path2):
        return True
    return False


def results_present_full_path(path, dataset, res):
    full_path = f"{path}/Predictions/{dataset}/testResample{res}.csv"
    full_path2 = f"{path}/Predictions/{dataset}/trainResample{res}.csv"
    if os.path.exists(full_path) and os.path.exists(full_path2):
        return True
    return False
