# -*- coding: utf-8 -*-
"""Clustering Experiments: code for experiments as an alternative to orchestration.

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

from tsml_eval.experiments import load_and_run_clustering_experiment
from tsml_eval.experiments.set_clusterer import set_clusterer
from tsml_eval.utils.experiments import _results_present, assign_gpu


def run_experiment(args, overwrite=False):
    """Mechanism for testing clusterers on the UCR data format.

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
        clusterer_name = args[3]
        dataset = args[4]
        resample = int(args[5])

        if len(args) > 6:
            test_fold = args[6].lower() == "false"
        else:
            test_fold = True

        if len(args) > 7:
            predefined_resample = args[7].lower() == "true"
        else:
            predefined_resample = False

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if not overwrite and _results_present(
            results_dir,
            clusterer_name,
            dataset,
            resample_id=resample,
            split="BOTH" if test_fold else "TRAIN",
        ):
            print("Ignoring, results already present")
        else:
            load_and_run_clustering_experiment(
                data_dir,
                results_dir,
                dataset,
                set_clusterer(clusterer_name, random_state=resample),
                resample_id=resample,
                clusterer_name=clusterer_name,
                overwrite=overwrite,
                build_test_file=test_fold,
                predefined_resample=predefined_resample,
            )
    # local run (no args)
    else:
        # These are example parameters, change as required for local runs
        # Do not include paths to your local directories here in PRs
        # If threading is required, see the threaded version of this file
        data_dir = "../"
        results_dir = "../"
        clusterer_name = "KMeans-DTW"
        dataset = "ArrowHead"
        resample = 0
        test_fold = False
        predefined_resample = False
        n_jobs = 4
        clusterer = set_clusterer(clusterer_name, random_state=resample, n_jobs=n_jobs)
        print(f"Local Run of {clusterer_name} ({clusterer.__class__.__name__}).")

        load_and_run_clustering_experiment(
            data_dir,
            results_dir,
            dataset,
            clusterer,
            resample_id=resample,
            clusterer_name=clusterer_name,
            overwrite=overwrite,
            build_test_file=test_fold,
            predefined_resample=predefined_resample,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
