"""TSER expansion publication tests."""

__all__ = ["_TSER_ARCHIVE_TEST_RESULTS_PATH"]

import os

_TSER_ARCHIVE_TEST_RESULTS_PATH = (
    "./test_output/expansion_regression/"
    if os.getcwd().split("\\")[-1] != "tests"
    else "../../../../../test_output/expansion_regression/"
)
