"""Test file checking that the RIST pipeline experiments run correctly."""

import os
import runpy

from tsml_eval.publications.y2023.rist_pipeline import (
    _run_classification_experiment,
    _run_regression_experiment,
)
from tsml_eval.publications.y2023.rist_pipeline.tests import _RIST_TEST_RESULTS_PATH
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH
from tsml_eval.utils.tests.test_results_writing import (
    _check_classification_file_format,
    _check_regression_file_format,
)


def test_run_rist_pipeline_classification_experiment():
    """Test paper classification experiments with test data and classifier."""
    classifier = "ROCKET"
    dataset = "MinimalChinatown"
    resample = 0

    args = [
        _TEST_DATA_PATH,
        _RIST_TEST_RESULTS_PATH,
        classifier,
        dataset,
        resample,
        "-ow",
    ]

    _run_classification_experiment(args)

    test_file = (
        f"{_RIST_TEST_RESULTS_PATH}{classifier}/Predictions/{dataset}/testResample0.csv"
    )
    assert os.path.exists(test_file)
    _check_classification_file_format(test_file)

    # this covers both the main method and present result file checking
    runpy.run_path(
        (
            "./tsml_eval/publications/y2023/rist_pipeline/"
            "run_classification_experiments.py"
            if os.getcwd().split("\\")[-1] != "tests"
            else "../run_classification_experiments.py"
        ),
        run_name="__main__",
    )

    os.remove(test_file)


def test_run_rist_pipeline_regression_experiment():
    """Test paper regression experiments with test data and regressor."""
    regressor = "ROCKET"
    dataset = "MinimalGasPrices"
    resample = 0

    args = [
        _TEST_DATA_PATH,
        _RIST_TEST_RESULTS_PATH,
        regressor,
        dataset,
        resample,
        "-ow",
    ]

    _run_regression_experiment(args)

    test_file = (
        f"{_RIST_TEST_RESULTS_PATH}{regressor}/Predictions/{dataset}/testResample0.csv"
    )
    assert os.path.exists(test_file)
    _check_regression_file_format(test_file)

    # this covers both the main method and present result file checking
    runpy.run_path(
        (
            "./tsml_eval/publications/y2023/rist_pipeline/run_regression_experiments.py"
            if os.getcwd().split("\\")[-1] != "tests"
            else "../run_regression_experiments.py"
        ),
        run_name="__main__",
    )

    os.remove(test_file)
