"""Test file checking that the TSC bakeoff experiments run correctly."""
import os

from tsml_eval.publications.y2023.tsc_bakeoff import _run_experiment
from tsml_eval.utils.tests.test_results_writing import _check_classification_file_format


def test_run_tsc_bakeoff_experiment():
    """Test paper classification experiments with test data and classifier."""
    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../../datasets/"
    )
    result_path = (
        "./test_output/tsc_bakeoff/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../../../test_output/tsc_bakeoff/"
    )
    classifier = "ROCKET"
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

    _run_experiment(args, overwrite=True, predefined_resample=False)

    test_file = f"{result_path}{classifier}/Predictions/{dataset}/testResample0.csv"
    assert os.path.exists(test_file)
    _check_classification_file_format(test_file)

    os.remove(test_file)
