"""Tests for the results validation utilities."""

import pytest

from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH
from tsml_eval.utils.results_validation import validate_results_file


@pytest.mark.parametrize(
    "path",
    [
        _TEST_RESULTS_PATH + "/regression/regressionResultsFile.csv",
        _TEST_RESULTS_PATH + "/classification/classificationResultsFile1.csv",
    ],
)
def test_validate_results_file(path):
    """Test results file validation with valid files."""
    assert validate_results_file(path)


@pytest.mark.parametrize(
    "path",
    [
        _TEST_RESULTS_PATH + "/broken/brokenRegressionResultsFile.csv",
        _TEST_RESULTS_PATH + "/broken/brokenClassificationResultsFile.csv",
        _TEST_RESULTS_PATH + "/broken/brokenResultsFile.csv",
    ],
)
def test_validate_broken_results_file(path):
    """Test results file validation with broken files."""
    assert not validate_results_file(path)
