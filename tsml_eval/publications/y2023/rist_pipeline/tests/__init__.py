"""RIST pipeline publication tests."""

__all__ = ["_RIST_TEST_RESULTS_PATH"]

import os
from pathlib import Path

_RIST_TEST_RESULTS_PATH = (
    os.path.dirname(Path(__file__).parent.parent.parent.parent.parent)
    + "/test_output/rist_pipeline/"
)
