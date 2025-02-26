"""Utilities for datasets."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "load_experiment_data",
    "copy_dataset_ts_files",
    "save_merged_dataset_splits",
]

import os
import shutil
from os.path import exists
from typing import Optional, Union

import numpy as np
from aeon.datasets import load_from_ts_file, write_to_ts_file


def load_experiment_data(
    problem_path: str,
    dataset: str,
    resample_id: int,
    predefined_resample: bool,
):
    """Load data for experiments.

    Parameters
    ----------
    problem_path : str
        Path to the problem folder.
    dataset : str
        Name of the dataset.
    resample_id : int or None
        Id of the data resample to use.
    predefined_resample : boolean
        If True, use the predefined resample.

    Returns
    -------
    X_train : np.ndarray or list of np.ndarray
        Train data in a 2d or 3d ndarray or list of arrays.
    y_train : np.ndarray
        Train data labels.
    X_test : np.ndarray or list of np.ndarray
        Test data in a 2d or 3d ndarray or list of arrays.
    y_test : np.ndarray
        Test data labels.
    resample : boolean
        If True, the data is to be resampled.
    """
    if resample_id is not None and predefined_resample:
        resample_str = "" if resample_id is None else str(resample_id)

        X_train, y_train = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TRAIN.ts"
        )
        X_test, y_test = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TEST.ts"
        )

        resample_data = False
    else:
        X_train, y_train = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}_TRAIN.ts"
        )
        X_test, y_test = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}_TEST.ts"
        )

        resample_data = True if resample_id != 0 else False

    return X_train, y_train, X_test, y_test, resample_data


def copy_dataset_ts_files(
    datasets: Union[list[str], str],
    source_path: str,
    destination_path: str,
):
    """Copy the TRAIN and TEST .ts files of the datasets to the destination path.

    Expects files to be present at {path}/{dataset}/{dataset}_TRAIN.ts and
    {path}/{dataset}/{dataset}_TEST.ts.

    Parameters
    ----------
    datasets : list of str or str
        The names of the datasets to copy. If a list, each item is the name of a
        dataset. If a string, it is the path to a file containing the names of the
        datasets, one per line.
    source_path : str
        The path to the directory containing the datasets.
    destination_path : str
        The path to the directory where the datasets will be copied.

    Examples
    --------
    >>> from tsml_eval.testing.testing_utils import _TEST_DATA_PATH, _TEST_OUTPUT_PATH
    >>> from tsml_eval.utils.datasets import copy_dataset_ts_files
    >>> copy_dataset_ts_files(
    ...     f"{_TEST_DATA_PATH}/_test_data/test_datalist.txt",
    ...     _TEST_DATA_PATH,
    ...     f"{_TEST_OUTPUT_PATH}/datasets/",
    ... )
    """
    if isinstance(datasets, str):
        with open(datasets) as f:
            datasets = f.readlines()
            datasets = [d.strip() for d in datasets]

    for file in datasets:
        data_file = f"{source_path}/{file}/{file}"
        os.makedirs(f"{destination_path}/{file}", exist_ok=True)

        if not exists(f"{data_file}_TRAIN.ts"):
            raise FileNotFoundError(f"File not found: {data_file}_TRAIN.ts")
        shutil.copy(
            f"{data_file}_TRAIN.ts", f"{destination_path}/{file}/{file}_TRAIN.ts"
        )

        if not exists(f"{data_file}_TEST.ts"):
            raise FileNotFoundError(f"File not found: {data_file}_TEST.ts")
        shutil.copy(f"{data_file}_TEST.ts", f"{destination_path}/{file}/{file}_TEST.ts")


def save_merged_dataset_splits(
    problem_path: str,
    dataset: str,
    save_path: Optional[str] = None,
):
    """Merge the TRAIN and TEST .ts files of a dataset and save the merged file.

    Expects files to be present at {path}/{dataset}/{dataset}_TRAIN.ts and
    {path}/{dataset}/{dataset}_TEST.ts.

    Parameters
    ----------
    problem_path : str
        Path to the problem folder.
    dataset : str
        Name of the dataset.
    save_path : str, default=None
        Path to save the merged dataset to. If None, the merged dataset will be saved
        in the same folder as the original datasets.
    """
    if save_path is None:
        save_path = problem_path

    X_train, y_train = load_from_ts_file(f"{problem_path}/{dataset}/{dataset}_TRAIN.ts")
    X_test, y_test = load_from_ts_file(f"{problem_path}/{dataset}/{dataset}_TEST.ts")

    os.makedirs(save_path, exist_ok=True)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    write_to_ts_file(X, f"{save_path}/{dataset}/", y=y, problem_name=dataset)
