"""Tests for the results validation and repair utilities."""

import os

import pytest

from tsml_eval.utils.experiments import fix_broken_second_line, validate_results_file
from tsml_eval.utils.test_utils import _TEST_DATA_PATH


@pytest.mark.parametrize(
    "path",
    [
        "test_files/regressionResultsFile.csv",
        "test_files/classificationResultsFile1.csv",
    ],
)
def test_validate_results_file(path):
    """Test results file validation with valid files."""
    path = (
        f"tsml_eval/utils/tests/{path}"
        if os.getcwd().split("\\")[-1] != "tests"
        else path
    )

    assert validate_results_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "test_files/brokenRegressionResultsFile.csv",
        "test_files/brokenClassificationResultsFile.csv",
        "test_files/brokenResultsFile.csv",
    ],
)
def test_validate_broken_results_file(path):
    """Test results file validation with broken files."""
    path = (
        f"tsml_eval/utils/tests/{path}"
        if os.getcwd().split("\\")[-1] != "tests"
        else path
    )

    assert not validate_results_file(path)


@pytest.mark.parametrize(
    "path",
    [
        ["test_files/regressionResultsFile.csv", 1],
        ["test_files/brokenRegressionResultsFile.csv", 2],
    ],
)
def test_fix_broken_second_line(path):
    """Test that the second line of a broken results file is fixed."""
    path[0] = (
        f"tsml_eval/utils/tests/{path[0]}"
        if os.getcwd().split("\\")[-1] != "tests"
        else path[0]
    )

    fix_broken_second_line(path[0], f"{_TEST_DATA_PATH}/secondLineTest{path[1]}.csv")

    assert validate_results_file(f"{_TEST_DATA_PATH}/secondLineTest{path[1]}.csv")

    # validate again while overwriting the original file
    fix_broken_second_line(f"{_TEST_DATA_PATH}/secondLineTest{path[1]}.csv")

    assert validate_results_file(f"{_TEST_DATA_PATH}/secondLineTest{path[1]}.csv")

    os.remove(f"{_TEST_DATA_PATH}/secondLineTest{path[1]}.csv")


def test_fix_broken_second_line_invalid_third_line():
    """Test that an error is raised if the third line is broken."""
    path = (
        "tsml_eval/utils/tests/test_files/brokenResultsFileLine3.csv"
        if os.getcwd().split("\\")[-1] != "tests"
        else "test_files/brokenResultsFileLine3.csv"
    )

    with pytest.raises(ValueError, match="No valid third line"):
        fix_broken_second_line(path, f"{_TEST_DATA_PATH}/secondLineTest3.csv")
