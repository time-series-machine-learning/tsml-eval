# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format. The capability for
threading is introduced.
"""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from set_classifier import set_classifier
from sktime.benchmarking.experiments import load_and_run_classification_experiment

from tsml_eval.utils.experiments import results_present


def run_experiment(args):
    """Mechanism for testing classifiers on the UCR format.

    This mirrors the mechanism used in the Java based tsml, but is not yet as
    engineered. Results generated using the method are in the same format as tsml and
    can be directly compared to the results generated in Java.
    """
    # cluster run (with args), this is fragile
    # don't run threaded jobs on the cluster unless you have reserved and node and know
    # what you are doing
    if args.__len__() > 1:
        print("Input args = ", sys.argv)
        data_dir = sys.argv[1]
        results_dir = sys.argv[2]
        classifier = sys.argv[3]
        dataset = sys.argv[4]
        # ADA starts indexing its jobs at 1, so we need to subtract 1
        resample = int(sys.argv[5]) - 1
        n_jobs = int(sys.argv[6])

        if len(sys.argv) > 7:
            train_fold = sys.argv[7].lower() == "true"
        else:
            train_fold = False

        if len(sys.argv) > 8:
            predefined_resample = sys.argv[8].lower() == "true"
        else:
            predefined_resample = False

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if results_present(results_dir, classifier, dataset, resample):
            print("Ignoring, results already present")
        else:
            load_and_run_classification_experiment(
                problem_path=data_dir,
                results_path=results_dir,
                classifier=set_classifier(classifier, resample, train_fold, n_jobs),
                cls_name=classifier,
                dataset=dataset,
                resample_id=resample,
                build_train=train_fold,
                predefined_resample=predefined_resample,
            )
    # local run (no args)
    else:
        data_dir = "../"
        results_dir = "../"
        cls_name = "DrCIF"
        dataset = "ItalyPowerDemand"
        resample = 0
        train_fold = False
        predefined_resample = False
        n_jobs = 4
        classifier = set_classifier(cls_name, resample, train_fold, n_jobs)
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
