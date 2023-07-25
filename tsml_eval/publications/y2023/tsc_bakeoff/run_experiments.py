"""Experiment runner for bakeoff redux 2023 publication."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os
import sys

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

from tsml_eval.experiments import load_and_run_classification_experiment
from tsml_eval.publications.y2023.tsc_bakeoff.set_bakeoff_classifier import (
    _set_bakeoff_classifier,
)
from tsml_eval.utils.experiments import _results_present

# all classifiers ran without duplicates
distance_based = ["1NN-DTW", "ShapeDTW"]
feature_based = ["Catch22", "FreshPRINCE", "TSFresh", "Signatures"]
shapelet_based = ["STC", "RDST", "RSF", "MrSQM"]
interval_based = ["R-STSF", "RISE", "TSF", "CIF", "STSF", "DrCIF"]
dictionary_based = ["BOSS", "cBOSS", "TDE", "WEASEL", "WEASEL-D"]
convolution_based = [
    "ROCKET",
    "MiniROCKET",
    "MultiROCKET",
    "Arsenal",
    "Hydra",
    "Hydra-MultiROCKET",
]
deep_learning = ["CNN", "ResNet", "InceptionTime"]
hybrid = ["HC1", "HC2"]
# top performing classifiers
top_classifiers = [
    "FreshPRINCE",
    "RDST",
    "R-STSF",
    "WEASEL-D",
    "Hydra-MultiROCKET",
    "InceptionTime",
    "HC2",
]


def _run_experiment(args, overwrite, predefined_resample):
    if args is None or args.__len__() <= 1:
        data_dir = "../"
        results_dir = "../"
        classifier_name = "ROCKET"
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
            _set_bakeoff_classifier(
                classifier_name,
                random_state=resample,
            ),
            resample_id=resample,
            classifier_name=classifier_name,
            overwrite=overwrite,
            predefined_resample=predefined_resample,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    args = sys.argv
    overwrite = True
    # The for the UCR 112 datasets set this to true to exactly reproduce results in the
    # paper. This uses the randomly generated resamples from tsml-java if the file is
    # present (see notebook for link). Except for PF, the 30 new datasets use
    # tsml-eval resamples.
    predefined_resample = False
    _run_experiment(args, overwrite, predefined_resample)
