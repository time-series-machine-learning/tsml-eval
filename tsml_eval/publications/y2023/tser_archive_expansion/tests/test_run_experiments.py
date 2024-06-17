"""Test file checking that the expansion regression experiments run correctly."""

import os
import runpy

from tsml_eval.publications.y2023.tser_archive_expansion import _run_experiment
from tsml_eval.publications.y2023.tser_archive_expansion.tests import (
    _TSER_ARCHIVE_TEST_RESULTS_PATH,
)
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH
from tsml_eval.utils.tests.test_results_writing import _check_regression_file_format


def test_run_expansion_regression_experiment():
    """Test paper regression experiments with test data and regressor."""
    regressor = "ROCKET"
    dataset = "MinimalGasPrices"
    resample = 0

    args = [
        _TEST_DATA_PATH,
        _TSER_ARCHIVE_TEST_RESULTS_PATH,
        regressor,
        dataset,
        resample,
        "-ow",
    ]

    _run_experiment(args)

    test_file = (
        f"{_TSER_ARCHIVE_TEST_RESULTS_PATH}{regressor}/Predictions/{dataset}/"
        "testResample0.csv"
    )
    assert os.path.exists(test_file)
    _check_regression_file_format(test_file)

    # this covers both the main method and present result file checking
    runpy.run_path(
        (
            "./tsml_eval/publications/y2023/tser_archive_expansion/run_experiments.py"
            if os.getcwd().split("\\")[-1] != "tests"
            else "../run_experiments.py"
        ),
        run_name="__main__",
    )

    os.remove(test_file)
