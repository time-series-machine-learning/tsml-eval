# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numba
import torch
from sktime.benchmarking.experiments import load_and_run_clustering_experiment

from tsml_eval.experiments.classification_experiments import results_present
from tsml_eval.experiments.set_clusterer import set_clusterer


def run_experiment(args, overwrite=False):
    numba.set_num_threads(1)
    torch.set_num_threads(1)

    # cluster run (with args), this is fragile
    if args.__len__() > 1:
        print("Input args = ", args)
        data_dir = args[1]
        results_dir = args[2]
        clusterer = args[3]
        dataset = args[4]
        # ADA starts indexing its jobs at 1, so we need to subtract 1
        resample = int(args[5]) - 1

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if not overwrite and results_present(results_dir, clusterer, dataset, resample):
            print("Ignoring, results already present")
        else:
            load_and_run_clustering_experiment(
                problem_path=data_dir,
                results_path=results_dir,
                clusterer=set_clusterer(clusterer, resample),
                cls_name=clusterer,
                dataset=dataset,
                resample_id=resample,
                train_file=True,
                overwrite=overwrite,
            )
    else:
        print("No local cluster experiment set up currently.")


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
