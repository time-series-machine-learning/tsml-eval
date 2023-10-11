"""Distance-based clustering tests."""

__all__ = ["_DISTANCE_TEST_RESULTS_PATH"]

import os
from pathlib import Path

_DISTANCE_TEST_RESULTS_PATH = (
    os.path.dirname(Path(__file__).parent.parent.parent.parent.parent)
    + "/test_output/distance_clustering/"
)
