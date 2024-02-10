"""Utilities for publications."""

import os
import shutil


def extract_publication_csvs_from_evaluation(stats, eval_path, write_path):
    """Extract the CSV files from the evaluation directory to a new directory.

    Parameters
    ----------
    stats : str or list of str
        The statistics to extract. The name of a statistic or list of names.
    eval_path : str or list of str
        The path to the evaluation directory generated from the evaluation package.
        If a list, each item is the path to an evaluation directory.
    write_path : str or list of str
        The path to the directory where the CSV files will be written. If a list,
        each item is the path to a directory where the CSV files will be written.
        eval_path and write_path must be the same length.

    Examples
    --------
    >>> from tsml_eval.testing.testing_utils import (
    ...     _TEST_EVAL_PATH, _TEST_OUTPUT_PATH
    ... )
    >>> from tsml_eval.utils.publications import (
    ...     extract_publication_csvs_from_evaluation
    ... )
    >>> extract_publication_csvs_from_evaluation(
    ...     ["Accuracy", "BalAcc"],
    ...     f"{_TEST_EVAL_PATH}/classification/",
    ...     f"{_TEST_OUTPUT_PATH}/result_files/",
    ... )
    """
    if isinstance(stats, str):
        stats = [stats]

    os.makedirs(write_path, exist_ok=True)

    for stat in stats:
        stat_dir = f"{eval_path}/{stat}/all_resamples/"

        for file in os.listdir(stat_dir):
            if file.endswith(".csv"):
                shutil.copy(f"{stat_dir}/{file}", f"{write_path}/{file}")
