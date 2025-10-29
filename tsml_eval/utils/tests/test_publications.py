"""Test publication utilities."""

import os

from tsml_eval.experiments import get_classifier_by_name
from tsml_eval.testing.testing_utils import _TEST_EVAL_PATH, _TEST_OUTPUT_PATH
from tsml_eval.utils.publications import (
    extract_publication_csv_from_evaluation,
    parameter_table_from_estimator_selector,
    results_table_from_evaluation_csv,
)


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


def test_parameter_table_from_estimator_selector():
    """Test creating a parameter table from an estimator selector."""
    table = parameter_table_from_estimator_selector(
        get_classifier_by_name, ["ROCKET", "TSF", "1NN-DTW"]
    )
    assert isinstance(table, str)


def test_results_table_from_evaluation_csv():
    """Test creating a results table from evaluation CSV files."""
    table = results_table_from_evaluation_csv(
        f"{_TEST_EVAL_PATH}/classification/Accuracy/accuracy_mean.csv"
    )
    assert isinstance(table, str)
    table2 = results_table_from_evaluation_csv(
        f"{_TEST_EVAL_PATH}/classification/Accuracy/accuracy_mean.csv",
        bold_best=False,
        round_digits=6,
        rank_columns=True,
    )
    assert isinstance(table2, str)
    table3 = results_table_from_evaluation_csv(
        f"{_TEST_EVAL_PATH}/classification/LogLoss/logloss_mean.csv",
        bold_best=True,
        round_digits=3,
        rank_columns=True,
        higher_is_better=False,
    )
    assert isinstance(table3, str)
