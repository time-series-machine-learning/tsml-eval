# -*- coding: utf-8 -*-
"""Regression Experiments: code for experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard tsml format.
"""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys

import numba
import numpy as np

from tsml_eval.experiments import run_forecasting_experiment
from tsml_eval.publications.forecasting23._set_forecaster import _set_forecaster
from tsml_eval.utils.experiments import _results_present, assign_gpu


def run_experiment(args, overwrite=False):
    """Mechanism for testing regressors on the UCR data format.

    This mirrors the mechanism used in the Java based tsml. Results generated using the
    method are in the same format as tsml and can be directly compared to the results
    generated in Java.
    """
    numba.set_num_threads(1)

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        try:
            gpu = assign_gpu()
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"Assigned GPU {gpu} to process.")
        except Exception:
            print("Unable to assign GPU to process.")

    # cluster run (with args), this is fragile
    if args is not None and args.__len__() > 1:
        print("Input args = ", args)
        data_dir = args[1]
        results_dir = args[2]
        forecaster_name = args[3]
        dataset = args[4]
        series_number = int(args[5])
    else:
        # https://github.com/Mcompetitions/M4-methods/tree/master/Dataset
        # Set these variables:
        data_dir = "D:/CMP Machine Learning/Datasets/Forecasting/M4/"
        results_dir = "D:/CMP Machine Learning/Datasets/Forecasting/M4Results/"
        forecaster_name = "DrCIF"
        dataset = "Daily"  # Hourly, Daily, Weekly, Monthly, Quarterly, Yearly
        series_number = 0  # Index from 0, skips header

    window_lengths = {
        "Hourly": 24,
        "Daily": 3,
        "Weekly": 3,
        "Monthly": 12,
        "Quarterly": 4,
        "Yearly": 3,
    }

    try:
        forecaster = _set_forecaster(
            forecaster_name,
            random_state=0,
            n_jobs=1,
            window_length=window_lengths[dataset],
        )
    except KeyError:
        raise ValueError(f"Unknown dataset {dataset}")

    print(
        f"Run of {forecaster_name} ({forecaster.__class__.__name__}) "
        f"on {dataset} series {series_number}."
    )

    series_name = f"{dataset}{series_number}"
    if not overwrite and _results_present(
        results_dir, forecaster_name, series_name, resample_id=None, split="TEST"
    ):
        print("Ignoring, results already present")
    else:
        train = None
        with open(f"{data_dir}/Train/{dataset}-train.csv", "r") as f:
            for i, line in enumerate(f):
                if i == series_number + 1:
                    train = np.array(
                        [
                            float(v)
                            for v in line.replace('"', "").rstrip(", \n").split(",")[1:]
                        ],
                        dtype=float,
                    )
                    break

        if train is None:
            raise IOError(
                f"Unable to read M4 dataset {dataset} train series {series_number}"
            )

        test = None
        with open(f"{data_dir}/Test/{dataset}-test.csv", "r") as f:
            for i, line in enumerate(f):
                if i == series_number + 1:
                    test = np.array(
                        [
                            float(v)
                            for v in line.replace('"', "").rstrip(", \n").split(",")[1:]
                        ],
                        dtype=float,
                    )
                    break

        if test is None:
            raise IOError(
                f"Unable to read M4 dataset {dataset} test series {series_number}"
            )

        run_forecasting_experiment(
            train,
            test,
            forecaster,
            results_dir,
            forecaster_name=forecaster_name,
            dataset_name=series_name,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv, overwrite=False)
