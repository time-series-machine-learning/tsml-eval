"""Test file checking that the expansion regression experiments run correctly."""
import os

from tsml_eval.publications.y2023.tser_archive_expansion import _run_experiment
from tsml_eval.utils.tests.test_results_writing import _check_regression_file_format


def test_run_expansion_regression_experiment():
    """Test paper regression experiments with test data and regressor."""
    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../../datasets/"
    )
    result_path = (
        "./test_output/expansion_regression/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../../../test_output/expansion_regression/"
    )
    regressor = "RandomForest"
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

    _run_experiment(args, overwrite=True)

    test_file = f"{result_path}{regressor}/Predictions/{dataset}/testResample0.csv"
    assert os.path.exists(test_file)
    _check_regression_file_format(test_file)

    os.remove(test_file)
