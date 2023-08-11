"""Test file checking that the RIST pipeline experiments run correctly."""

import os

from tsml_eval.publications.y2023.rist_pipeline import (
    _run_classification_experiment,
    _run_regression_experiment,
)
from tsml_eval.utils.tests.test_results_writing import (
    _check_classification_file_format,
    _check_regression_file_format,
)


def test_run_rist_pipeline_classification_experiment():
    """Test paper classification experiments with test data and classifier."""
    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../../datasets/"
    )
    result_path = (
        "./test_output/tsc_bakeoff/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../../../test_output/rist_pipeline/"
    )
    classifier = "RIST"
    dataset = "MinimalChinatown"
    resample = 0

    args = [
        None,
        data_path,
        result_path,
        classifier,
        dataset,
        resample,
    ]

    _run_classification_experiment(args, overwrite=True)

    test_file = f"{result_path}{classifier}/Predictions/{dataset}/testResample0.csv"
    assert os.path.exists(test_file)
    _check_classification_file_format(test_file)

    os.remove(test_file)


def test_run_rist_pipeline_regression_experiment():
    """Test paper regression experiments with test data and regressor."""
    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../../datasets/"
    )
    result_path = (
        "./test_output/tsc_bakeoff/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../../../test_output/rist_pipeline/"
    )
    regressor = "RIST"
    dataset = "MinimalGasPrices"
    resample = 0

    args = [
        None,
        data_path,
        result_path,
        regressor,
        dataset,
        resample,
    ]

    _run_regression_experiment(args, overwrite=True)

    test_file = f"{result_path}{regressor}/Predictions/{dataset}/testResample0.csv"
    assert os.path.exists(test_file)
    _check_regression_file_format(test_file)

    os.remove(test_file)
