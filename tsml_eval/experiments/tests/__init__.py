"""Experiments tests."""

__all__ = [
    "_CLASSIFIER_RESULTS_PATH",
    "_CLUSTERER_RESULTS_PATH",
    "_REGRESSOR_RESULTS_PATH",
]

import os

_CLASSIFIER_RESULTS_PATH = (
    "./test_output/classification/"
    if os.getcwd().split("\\")[-1] != "tests"
    else "../../../test_output/classification/"
)

_CLUSTERER_RESULTS_PATH = (
    "./test_output/clustering/"
    if os.getcwd().split("\\")[-1] != "tests"
    else "../../../test_output/clustering/"
)

_REGRESSOR_RESULTS_PATH = (
    "./test_output/regression/"
    if os.getcwd().split("\\")[-1] != "tests"
    else "../../../test_output/regression/"
)
