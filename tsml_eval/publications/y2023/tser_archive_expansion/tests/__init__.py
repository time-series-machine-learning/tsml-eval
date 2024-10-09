"""TSER expansion publication tests."""

__all__ = ["_TSER_ARCHIVE_TEST_RESULTS_PATH"]

import os
from pathlib import Path

_TSER_ARCHIVE_TEST_RESULTS_PATH = (
    os.path.dirname(Path(__file__).parent.parent.parent.parent.parent)
    + "/test_output/expansion_regression/"
)
