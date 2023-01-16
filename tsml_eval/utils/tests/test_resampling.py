# -*- coding: utf-8 -*-
"""Tests for dataset resampling functions."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os

import numpy as np
import pandas as pd
import pytest
from sktime.datasets import load_arrow_head

from tsml_eval.utils.experiments import (
    compare_result_file_resample,
    resample_data,
    stratified_resample_data,
)


def test_resample_data():
    """Test resampling returns valid data."""
    X_train, y_train = load_arrow_head(split="TRAIN")
    X_test, y_test = load_arrow_head(split="TEST")

    train_size = X_train.shape
    test_size = X_test.shape

    X_train, y_train, X_test, y_test = resample_data(X_train, y_train, X_test, y_test)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_test, pd.np.ndarray)

    assert X_train.shape == train_size
    assert X_test.shape == test_size


def test_stratified_resample_data():
    """Test stratified resampling returns valid data and class distribution."""
    X_train, y_train = load_arrow_head(split="TRAIN")
    X_test, y_test = load_arrow_head(split="TEST")

    train_size = X_train.shape
    test_size = X_test.shape
    _, counts_train = np.unique(y_train, return_counts=True)
    _, counts_test = np.unique(y_test, return_counts=True)

    X_train, y_train, X_test, y_test = stratified_resample_data(
        X_train, y_train, X_test, y_test
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_test, pd.np.ndarray)

    assert X_train.shape == train_size
    assert X_test.shape == test_size

    _, counts_train_new = np.unique(y_train, return_counts=True)
    _, counts_test_new = np.unique(y_test, return_counts=True)

    assert list(counts_train_new) == list(counts_train)
    assert list(counts_test_new) == list(counts_test)


@pytest.mark.parametrize(
    "paths",
    [
        [
            "test_files/classificationResultsFile1.csv",
            "test_files/classificationResultsFile1.csv",
            True,
        ],
        [
            "test_files/classificationResultsFile1.csv",
            "test_files/classificationResultsFile2.csv",
            False,
        ],
    ],
)
def test_compare_result_file_resample(paths):
    """Test compare result file resample function."""
    if os.getcwd().split("\\")[-1] != "tests":
        paths[0] = f"tsml_eval/utils/tests/{paths[0]}"
        paths[1] = f"tsml_eval/utils/tests/{paths[1]}"

    assert compare_result_file_resample(paths[0], paths[1]) == paths[2]
