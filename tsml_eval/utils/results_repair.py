"""Utility functions for repairing results."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["fix_broken_second_line"]

import os

from tsml_eval.utils.results_validation import (
    _check_classification_third_line,
    _check_clustering_third_line,
    _check_forecasting_third_line,
    _check_regression_third_line,
)


def fix_broken_second_line(file_path, save_path=None):
    """Fix a results while where the written second line has line breaks.

    This function will remove line breaks from any lines between the first line and the
    first seen valid 'third_line' for any results file format.

    Parameters
    ----------
    file_path : str
        Path to the results file to be fixed, including the file itself.
    save_path : str, default=None
        Path to save the fixed results file to, including the file new files name.
        If None, the new file will replace the original file.
    """
    with open(file_path) as f:
        lines = f.readlines()

    line_count = 2
    while (
        not _check_classification_third_line(lines[line_count])
        and not _check_regression_third_line(lines[line_count])
        and not _check_clustering_third_line(lines[line_count])
        and not _check_forecasting_third_line(lines[line_count])
    ):
        if line_count == len(lines) - 1:
            raise ValueError("No valid third line found in input results file.")
        line_count += 1

    if line_count != 2:
        lines[1] = lines[1].replace("\n", " ").replace("\r", " ")
        for i in range(2, line_count - 1):
            lines[1] = lines[1] + lines[i].replace("\n", " ").replace("\r", " ")
        lines[1] = lines[1] + lines[line_count - 1]
        lines = lines[:2] + lines[line_count:]

    if save_path is not None or line_count != 2:
        if save_path is None:
            save_path = file_path

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            f.writelines(lines)
