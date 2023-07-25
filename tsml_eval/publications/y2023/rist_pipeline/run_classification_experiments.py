"""Experiment runner for RIST pipeline publication."""

__author__ = ["MatthewMiddlehurst"]

import os
import sys

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

from tsml_eval.experiments import load_and_run_classification_experiment
from tsml_eval.publications.y2023.rist_pipeline.set_rist_classifier import (
    _set_rist_classifier,
)
from tsml_eval.utils.experiments import _results_present

classifiers = [
    "FreshPRINCE",
    "STC",
    "RDST",
    "R-STSF",
    "DrCIF",
    "ROCKET",
    "HC2",
    "RIST-ExtraT",
    "RIST-RF",
    "RIST-RidgeCV",
    "IntervalPipeline",
]


def _run_classification_experiment(args, overwrite):
    if args is None or args.__len__() <= 1:
        data_dir = "../"
        results_dir = "../"
        classifier_name = "RIST"
        dataset = "ItalyPowerDemand"
        resample = 0
    else:
        print("Input args = ", args)
        # ignore args[0]
        data_dir = args[1]
        results_dir = args[2]
        classifier_name = args[3]
        dataset = args[4]
        resample = int(args[5])

    # Skip if not overwrite and results already present
    # this is also checked in load_and_run, but doing a quick check here so can
    # print a message and make sure data is not loaded
    if not overwrite and _results_present(
        results_dir,
        classifier_name,
        dataset,
        resample_id=resample,
        split="TEST",
    ):
        print("Ignoring, results already present")
    else:
        load_and_run_classification_experiment(
            data_dir,
            results_dir,
            dataset,
            _set_rist_classifier(
                classifier_name,
                random_state=resample,
            ),
            resample_id=resample,
            classifier_name=classifier_name,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    args = sys.argv
    overwrite = True
    _run_classification_experiment(args, overwrite)
