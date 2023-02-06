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

from tsml_eval.experiments import load_and_run_forecasting_experiment
from tsml_eval.experiments.set_forecaster import set_forecaster
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
        resample = int(args[5])

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if not overwrite and _results_present(
            results_dir, forecaster_name, dataset, resample
        ):
            print("Ignoring, results already present")
        else:
            load_and_run_forecasting_experiment(
                data_dir,
                results_dir,
                dataset,
                set_forecaster(
                    forecaster_name,
                    random_state=resample,
                ),
                resample_id=resample,
                regressor_name=forecaster_name,
                overwrite=overwrite,
            )
    # local run (no args)
    else:
        # These are example parameters, change as required for local runs
        # Do not include paths to your local directories here in PRs
        # If threading is required, see the threaded version of this file
        data_dir = "../"
        results_dir = "../"
        forecaster_name = "LR"
        dataset = "??"
        resample = 0
        forecaster = set_forecaster(
            forecaster_name,
            random_state=resample,
        )
        print(f"Local Run of {forecaster_name} ({forecaster.__class__.__name__}).")

        load_and_run_forecasting_experiment(
            data_dir,
            results_dir,
            dataset,
            forecaster,
            resample_id=resample,
            regressor_name=forecaster_name,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
