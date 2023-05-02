# -*- coding: utf-8 -*-
__author__ = ["TonyBagnall", "MatthewMiddlehurst", "dguijo"]

import sys

from tsml_eval.experiments import load_and_run_regression_experiment
from tsml_eval.publications._2023.tser_archive_expansion.set_expansion_regressor import (
    _set_expansion_regressor,
)
from tsml_eval.utils.experiments import _results_present

if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    args = sys.argv
    overwrite = True
    if args is None or args.__len__() <= 1:
        data_dir = "../"
        results_dir = "../"
        regressor_name = "LR"
        dataset = "Covid3Month"
        resample = 0
    else:
        print("Input args = ", args)
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
            _set_expansion_regressor(
                regressor_name,
                random_state=resample,
            ),
            resample_id=resample,
            regressor_name=regressor_name,
            overwrite=overwrite,
        )
