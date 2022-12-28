# -*- coding: utf-8 -*-
"""Regressor Experiments: code to run experiments and generate results file in
standard format.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format. It is cloned from
classification_experiments, we should condense it all to one.
"""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys

import numba
import torch

from tsml_eval.experiments.set_regressor import set_regressor
from tsml_eval.utils.experiments import _results_present


def run_experiment(args, overwrite=False):
    numba.set_num_threads(1)
    torch.set_num_threads(1)

    # cluster run (with args), this is fragile
    if args.__len__() > 1:  # cluster run, this is fragile
        print("Input args = ", args)
        data_dir = args[1]
        results_dir = args[2]
        regressor_name = args[3]
        dataset = args[4]
        # ADA starts indexing its jobs at 1, so we need to subtract 1
        resample = int(args[5]) - 1

        if len(args) > 6:
            train_fold = args[6].lower() == "true"
        else:
            train_fold = False

        if len(args) > 7:
            predefined_resample = args[7].lower() == "true"
        else:
            predefined_resample = False

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if not overwrite and _results_present(
            results_dir, regressor_name, dataset, resample
        ):
            print("Ignoring, results already present")
        else:
            load_and_run_regression_experiment(
                problem_path=data_dir,
                results_path=results_dir,
                regressor=set_regressor(regressor_name, resample, train_fold),
                regressor_name=regressor_name,
                dataset=dataset,
                resample_id=resample,
                build_train=train_fold,
                predefined_resample=predefined_resample,
                overwrite=overwrite,
            )
    # local run (no args)
    else:
        print(" Local Run of TimeSeriesForestRegressor")
        data_dir = "../../../time_series_regression/new_datasets/"
        results_dir = "../"
        regressor_name = "svr"
        dataset = "Covid3Months"
        resample = 0
        train_fold = False
        predefined_resample = False
        regressor = set_regressor(regressor_name)
        print(f"Local Run of {regressor.__class__.__name__}.")

        load_and_run_regression_experiment(
            problem_path=data_dir,
            results_path=results_dir,
            regressor=regressor,
            regressor_name=regressor_name,
            dataset=dataset,
            resample_id=resample,
            build_train=train_fold,
            predefined_resample=predefined_resample,
            overwrite=True,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
