"""Test dataset utilities."""

import os

import pytest
from aeon.datasets import load_from_ts_file

from tsml_eval.datasets._test_data._data_sizes import DATA_TEST_SIZES, DATA_TRAIN_SIZES
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH, _TEST_OUTPUT_PATH
from tsml_eval.utils.datasets import copy_dataset_ts_files, save_merged_dataset_splits


def test_copy_dataset_ts_files():
    """Test copying .ts dataset files."""
    copy_path = f"{_TEST_OUTPUT_PATH}/datasets/"

    copy_dataset_ts_files(
        f"{_TEST_DATA_PATH}/_test_data/test_datalist.txt",
        _TEST_DATA_PATH,
        copy_path,
    )

    assert os.path.exists(f"{copy_path}/MinimalChinatown/MinimalChinatown_TRAIN.ts")
    assert os.path.exists(f"{copy_path}/MinimalChinatown/MinimalChinatown_TEST.ts")
    assert os.path.exists(f"{copy_path}/MinimalGasPrices/MinimalGasPrices_TRAIN.ts")
    assert os.path.exists(f"{copy_path}/MinimalGasPrices/MinimalGasPrices_TEST.ts")
    assert os.path.exists(
        f"{_TEST_DATA_PATH}/MinimalChinatown/MinimalChinatown_TRAIN.ts"
    )
    assert os.path.exists(
        f"{_TEST_DATA_PATH}/MinimalChinatown/MinimalChinatown_TEST.ts"
    )
    assert os.path.exists(
        f"{_TEST_DATA_PATH}/MinimalGasPrices/MinimalGasPrices_TRAIN.ts"
    )
    assert os.path.exists(
        f"{_TEST_DATA_PATH}/MinimalGasPrices/MinimalGasPrices_TEST.ts"
    )

    os.remove(f"{copy_path}/MinimalChinatown/MinimalChinatown_TRAIN.ts")
    os.remove(f"{copy_path}/MinimalChinatown/MinimalChinatown_TEST.ts")
    os.remove(f"{copy_path}/MinimalGasPrices/MinimalGasPrices_TRAIN.ts")
    os.remove(f"{copy_path}/MinimalGasPrices/MinimalGasPrices_TEST.ts")


@pytest.mark.parametrize(
    "save_path", [None, f"{_TEST_OUTPUT_PATH}/datasets/merged2/destination/"]
)
def test_save_merged_dataset_splits(save_path):
    """Test merging and saving .ts dataset files."""
    copy_path = (
        f"{_TEST_OUTPUT_PATH}/datasets/merged/"
        if save_path is None
        else f"{_TEST_OUTPUT_PATH}/datasets/merged2/"
    )

    copy_dataset_ts_files(
        ["MinimalChinatown"],
        _TEST_DATA_PATH,
        copy_path,
    )

    save_merged_dataset_splits(
        copy_path,
        "MinimalChinatown",
        save_path,
    )

    if save_path is None:
        save_path = copy_path

    assert os.path.exists(f"{copy_path}/MinimalChinatown/MinimalChinatown_TRAIN.ts")
    assert os.path.exists(f"{copy_path}/MinimalChinatown/MinimalChinatown_TEST.ts")
    assert os.path.exists(f"{save_path}/MinimalChinatown/MinimalChinatown.ts")

    X, y = load_from_ts_file(f"{save_path}/MinimalChinatown/MinimalChinatown.ts")

    new_len = DATA_TRAIN_SIZES["MinimalChinatown"] + DATA_TEST_SIZES["MinimalChinatown"]
    assert X.shape[0] == new_len
    assert y.shape[0] == new_len

    os.remove(f"{copy_path}/MinimalChinatown/MinimalChinatown_TRAIN.ts")
    os.remove(f"{copy_path}/MinimalChinatown/MinimalChinatown_TEST.ts")
    os.remove(f"{save_path}/MinimalChinatown/MinimalChinatown.ts")
