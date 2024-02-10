"""Utilities for datasets."""

import os
import shutil
from os.path import exists
from typing import List, Union


def copy_dataset_ts_files(
    datasets: Union[List[str], str],
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
        with open(datasets, "r") as f:
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
