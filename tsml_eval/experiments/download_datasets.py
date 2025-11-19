import os
import sys

from aeon.datasets import load_forecasting


def check_and_download_datasets(args):
    data_path = args[0]
    dataset_file_path = args[1]
    with open(dataset_file_path, 'r') as dataset_file:
        lines = dataset_file.readlines()
        for dataset_name in lines:
            if "_" in dataset_name:
                dataset, _ = dataset_name.rsplit("_", 1)
            else:
                raise ValueError(f"Dataset {dataset_name} given, but has no series attached when remote_forecasting_experiment called.")
            data_full_path = f"{data_path}/{dataset}/{dataset}.tsf"
            if not os.path.exists(data_full_path):
                print(f"Downloading {dataset}")
                load_forecasting(dataset, data_path)

if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    print("Running download_datsets.py")
    check_and_download_datasets(sys.argv[1:])