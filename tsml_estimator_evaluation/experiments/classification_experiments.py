# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from sktime.benchmarking.experiments import load_and_run_classification_experiment

from tsml_estimator_evaluation.experiments.set_classifier import set_classifier
from tsml_estimator_evaluation.utils.experiments import results_present


def run_experiment(args, overwrite=False):
    """Mechanism for testing classifiers on the UCR format.

    This mirrors the mechanism used in the Java based tsml, but is not yet as
    engineered. Results generated using the method are in the same format as tsml and
    can be directly compared to the results generated in Java.
    """
    # cluster run (with args), this is fragile
    if args is not None and args.__len__() > 1:
        print("Input args = ", args)
        data_dir = args[1]
        results_dir = args[2]
        classifier = args[3]
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
        if not overwrite and results_present(
            results_dir, classifier, dataset, resample
        ):
            print("Ignoring, results already present")
        else:
            load_and_run_classification_experiment(
                problem_path=data_dir,
                results_path=results_dir,
                classifier=set_classifier(classifier, resample, train_fold),
                cls_name=classifier,
                dataset=dataset,
                resample_id=resample,
                build_train=train_fold,
                predefined_resample=predefined_resample,
                overwrite=overwrite,
            )
    else:  # Local run
        data_dir = "/home/ajb/Data/"
        results_dir = "/home/ajb/Results Working Area/ReduxBakeoff/sktime/"
        cls_name = "ROCKET"
        n_jobs = 90
        contract_mins = 0
        print(f" Local Run of {cls_name} with threading jobs ={ n_jobs} contract time ={contract_mins}")
        classifier = set_classifier(cls_name, n_jobs=n_jobs, contract=contract_mins)
        dataset = "Tiselac"
        resample = 0
        train_fold = True
        predefined_resample = False
        classifier = set_classifier(cls_name, resample, train_fold)
        print(f"Local Run of {classifier.__class__.__name__}.")

        load_and_run_classification_experiment(
            overwrite=False,
            problem_path=data_dir,
            results_path=results_dir,
            cls_name=cls_name,
            classifier=classifier,
            dataset=dataset,
            resample_id=resample,
            build_train=train_fold,
            predefined_resample=predefined_resample,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
