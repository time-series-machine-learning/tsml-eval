"""Bakeoff redux 2023 publication tests."""

__all__ = ["_BAKEOFF_TEST_RESULTS_PATH"]

import os
from pathlib import Path

_BAKEOFF_TEST_RESULTS_PATH = (
    os.path.dirname(Path(__file__).parent.parent.parent.parent.parent)
    + "/test_output/tsc_bakeoff/"
)
