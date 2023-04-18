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

from tsml_eval.experiments import load_and_run_regression_experiment
from tsml_eval.experiments.set_regressor import set_regressor
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
        regressor_name = args[3]
        dataset = args[4]
        resample = int(args[5])

        if len(args) > 6:
            train_fold = args[6].lower() == "true"
        else:
            train_fold = False

        if len(args) > 7:
            predefined_resample = args[7].lower() == "true"
        else:
            predefined_resample = False

        if len(args) > 8:
            checkpoint = f"{args[8]}{regressor_name}/{dataset}/{resample}"
        else:
            checkpoint = None

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if not overwrite and _results_present(
            results_dir,
            regressor_name,
            dataset,
            resample_id=resample,
            split="BOTH" if train_fold else "TEST",
        ):
            print("Ignoring, results already present")
        else:
            load_and_run_regression_experiment(
                data_dir,
                results_dir,
                dataset,
                set_regressor(
                    regressor_name,
                    random_state=resample,
                    build_train_file=train_fold,
                    checkpoint=checkpoint,
                ),
                resample_id=resample,
                regressor_name=regressor_name,
                overwrite=overwrite,
                build_train_file=train_fold,
                predefined_resample=predefined_resample,
            )
    # local run (no args)
    else:
        # These are example parameters, change as required for local runs
        # Do not include paths to your local directories here in PRs
        # If threading is required, see the threaded version of this file
        data_dir = "../"
        results_dir = "../"
        regressor_name = "LR"
        dataset = "Covid3Month"
        resample = 0
        checkpoint = None
        train_fold = False
        predefined_resample = False
        regressor = set_regressor(
            regressor_name,
            random_state=resample,
            build_train_file=train_fold,
            checkpoint=checkpoint,
        )
        print(f"Local Run of {regressor_name} ({regressor.__class__.__name__}).")

        load_and_run_regression_experiment(
            data_dir,
            results_dir,
            dataset,
            regressor,
            resample_id=resample,
            regressor_name=regressor_name,
            overwrite=overwrite,
            build_train_file=train_fold,
            predefined_resample=predefined_resample,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
