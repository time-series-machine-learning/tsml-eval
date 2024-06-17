"""Experiments tests."""

__all__ = [
    "_CLASSIFIER_RESULTS_PATH",
    "_CLUSTERER_RESULTS_PATH",
    "_FORECASTER_RESULTS_PATH",
    "_REGRESSOR_RESULTS_PATH",
]

from tsml_eval.testing.testing_utils import _TEST_OUTPUT_PATH

_CLASSIFIER_RESULTS_PATH = _TEST_OUTPUT_PATH + "/classification/"

_CLUSTERER_RESULTS_PATH = _TEST_OUTPUT_PATH + "/clustering/"

_FORECASTER_RESULTS_PATH = _TEST_OUTPUT_PATH + "/forecasting/"

_REGRESSOR_RESULTS_PATH = _TEST_OUTPUT_PATH + "/regression/"
