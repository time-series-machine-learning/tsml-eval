"""Experiments tests."""

__all__ = [
    "_CLASSIFIER_RESULTS_PATH",
    "_CLUSTERER_RESULTS_PATH",
    "_FORECASTER_RESULTS_PATH",
    "_REGRESSOR_RESULTS_PATH",
]

import os
from pathlib import Path

_CLASSIFIER_RESULTS_PATH = (
    os.path.dirname(Path(__file__).parent.parent) + "/test_output/classification/"
)

_CLUSTERER_RESULTS_PATH = (
    os.path.dirname(Path(__file__).parent.parent) + "/test_output/clustering/"
)

_FORECASTER_RESULTS_PATH = (
    os.path.dirname(Path(__file__).parent.parent) + "/test_output/forecasting/"
)

_REGRESSOR_RESULTS_PATH = (
    os.path.dirname(Path(__file__).parent.parent) + "/test_output/regression/"
)
