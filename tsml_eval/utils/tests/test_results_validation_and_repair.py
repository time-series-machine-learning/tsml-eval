"""Tests for the results validation and repair utilities."""

import os

import pytest

from tsml_eval.testing.test_utils import _TEST_OUTPUT_PATH, _TEST_RESULTS_PATH
from tsml_eval.utils.experiments import fix_broken_second_line
from tsml_eval.utils.validation import validate_results_file


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


@pytest.mark.parametrize(
    "path",
    [
        [_TEST_RESULTS_PATH + "/regression/regressionResultsFile.csv", 1],
        [_TEST_RESULTS_PATH + "/broken/brokenRegressionResultsFile.csv", 2],
    ],
)
def test_fix_broken_second_line(path):
    """Test that the second line of a broken results file is fixed."""
    fix_broken_second_line(path[0], f"{_TEST_OUTPUT_PATH}/secondLineTest{path[1]}.csv")

    assert validate_results_file(f"{_TEST_OUTPUT_PATH}/secondLineTest{path[1]}.csv")

    # validate again while overwriting the original file
    fix_broken_second_line(f"{_TEST_OUTPUT_PATH}/secondLineTest{path[1]}.csv")

    assert validate_results_file(f"{_TEST_OUTPUT_PATH}/secondLineTest{path[1]}.csv")

    os.remove(f"{_TEST_OUTPUT_PATH}/secondLineTest{path[1]}.csv")


def test_fix_broken_second_line_invalid_third_line():
    """Test that an error is raised if the third line is broken."""
    path = _TEST_RESULTS_PATH + "/broken/brokenResultsFileLine3.csv"

    with pytest.raises(ValueError, match="No valid third line"):
        fix_broken_second_line(path, f"{_TEST_OUTPUT_PATH}/secondLineTest3.csv")
