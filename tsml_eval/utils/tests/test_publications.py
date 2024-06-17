"""Test publication utilities."""

import os

from tsml_eval.testing.testing_utils import _TEST_EVAL_PATH, _TEST_OUTPUT_PATH
from tsml_eval.utils.publications import extract_publication_csv_from_evaluation


def test_extract_publication_csv_from_evaluation():
    """Test extracting publication results CSVs from evaluation directories."""
    extract_publication_csv_from_evaluation(
        "Accuracy",
        f"{_TEST_EVAL_PATH}/classification/",
        f"{_TEST_OUTPUT_PATH}/eval_result_files_test/",
    )

    assert os.path.exists(
        f"{_TEST_EVAL_PATH}/classification/Accuracy/all_resamples/1NN-DTW_accuracy.csv"
    )
    assert os.path.exists(
        f"{_TEST_EVAL_PATH}/classification/Accuracy/all_resamples/ROCKET_accuracy.csv"
    )
    assert os.path.exists(
        f"{_TEST_EVAL_PATH}/classification/Accuracy/all_resamples/TSF_accuracy.csv"
    )
    assert os.path.exists(
        f"{_TEST_OUTPUT_PATH}/eval_result_files_test/1NN-DTW_accuracy.csv"
    )
    assert os.path.exists(
        f"{_TEST_OUTPUT_PATH}/eval_result_files_test/ROCKET_accuracy.csv"
    )
    assert os.path.exists(
        f"{_TEST_OUTPUT_PATH}/eval_result_files_test/TSF_accuracy.csv"
    )

    os.remove(f"{_TEST_OUTPUT_PATH}/eval_result_files_test/1NN-DTW_accuracy.csv")
    os.remove(f"{_TEST_OUTPUT_PATH}/eval_result_files_test/ROCKET_accuracy.csv")
    os.remove(f"{_TEST_OUTPUT_PATH}/eval_result_files_test/TSF_accuracy.csv")
