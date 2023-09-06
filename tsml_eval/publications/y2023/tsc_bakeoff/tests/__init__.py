"""Bakeoff redux 2023 publication tests."""

__all__ = ["_BAKEOFF_TEST_RESULTS_PATH"]

import os

_BAKEOFF_TEST_RESULTS_PATH = (
    "./test_output/tsc_bakeoff/"
    if os.getcwd().split("\\")[-1] != "tests"
    else "../../../../../test_output/tsc_bakeoff/"
)
