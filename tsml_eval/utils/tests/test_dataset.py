import os

from tsml_eval.testing.testing_utils import _TEST_DATA_PATH, _TEST_OUTPUT_PATH
from tsml_eval.utils.datasets import copy_dataset_ts_files


def test_copy_dataset_ts_files():
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
