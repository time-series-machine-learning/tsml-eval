"""Test file checking that the TSC bakeoff experiments run correctly."""

import os
import runpy

from tsml_eval.publications.y2023.tsc_bakeoff.run_experiments import _run_experiment
from tsml_eval.publications.y2023.tsc_bakeoff.tests import _BAKEOFF_TEST_RESULTS_PATH
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH
from tsml_eval.utils.tests.test_results_writing import _check_classification_file_format


def test_run_tsc_bakeoff_experiment():
    """Test paper classification experiments with test data and classifier."""
    classifier = "ROCKET"
    dataset = "MinimalChinatown"
    resample = 0

    args = [
        _TEST_DATA_PATH,
        _BAKEOFF_TEST_RESULTS_PATH,
        classifier,
        dataset,
        resample,
        "-ow",
    ]

    _run_experiment(args, predefined_resample=False)

    test_file = (
        f"{_BAKEOFF_TEST_RESULTS_PATH}{classifier}/Predictions/{dataset}/"
        "testResample0.csv"
    )
    assert os.path.exists(test_file)
    _check_classification_file_format(test_file)

    # this covers both the main method and present result file checking
    runpy.run_path(
        (
            "./tsml_eval/publications/y2023/tsc_bakeoff/run_experiments.py"
            if os.getcwd().split("\\")[-1] != "tests"
            else "../run_experiments.py"
        ),
        run_name="__main__",
    )

    os.remove(test_file)
