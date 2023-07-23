"""Experiment runner for RIST pipeline publication."""

__author__ = ["MatthewMiddlehurst"]

import os
import sys

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

from tsml_eval.experiments import load_and_run_regression_experiment
from tsml_eval.publications.y2023.rist_pipeline.set_rist_regressor import (
    _set_rist_regressor,
)
from tsml_eval.utils.experiments import _results_present

regressors = [
    "InceptionTime",
    "ROCKET",
    "DrCIF",
    "FreshPRINCE",
    "RDST",
    "RIST-ExtraT",
    "RIST-RF",
    "RIST-RidgeCV",
]


def _run_regression_experiment(args, overwrite):
    if args is None or args.__len__() <= 1:
        data_dir = "../"
        results_dir = "../"
        regressor_name = "RIST"
        dataset = "Covid3Month"
        resample = 0
    else:
        print("Input args = ", args)
        # ignore args[0]
        data_dir = args[1]
        results_dir = args[2]
        regressor_name = args[3]
        dataset = args[4]
        resample = int(args[5])

    # Skip if not overwrite and results already present
    # this is also checked in load_and_run, but doing a quick check here so can
    # print a message and make sure data is not loaded
    if not overwrite and _results_present(
        results_dir,
        regressor_name,
        dataset,
        resample_id=resample,
        split="TEST",
    ):
        print("Ignoring, results already present")
    else:
        load_and_run_regression_experiment(
            data_dir,
            results_dir,
            dataset,
            _set_rist_regressor(
                regressor_name,
                random_state=resample,
            ),
            resample_id=resample,
            regressor_name=regressor_name,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    args = sys.argv
    overwrite = True
    _run_regression_experiment(args, overwrite)
