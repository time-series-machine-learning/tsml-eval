# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

__author__ = ["TonyBagnall"]

import os

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numba
import numpy as np
import torch
from sklearn.metrics import davies_bouldin_score
from sktime.benchmarking.experiments import run_clustering_experiment
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.datasets import load_from_tsfile as load_ts

from tsml_estimator_evaluation.experiments.classification_experiments import (
    results_present,
)


def config_clusterer(clusterer: str, **kwargs):
    """Config clusterer."""
    if clusterer == "kmeans":
        cls = TimeSeriesKMeans(**kwargs)
    elif clusterer == "kmedoids":
        cls = TimeSeriesKMedoids(**kwargs)
    return cls


def tune_window(metric: str, train_X, n_clusters):
    """Tune window."""
    best_w = 0
    best_score = 0
    for w in np.arange(0, 1, 0.1):
        cls = TimeSeriesKMeans(
            metric=metric, distance_params={"window": w}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        print(" Preds type = ", type(preds))
        score = davies_bouldin_score(train_X, preds)
        print(score)
        if score > best_score:
            best_score = score
            best_w = w
    print("best window =", best_w, " with score ", best_score)
    return best_w


def _recreate_results(trainX, trainY):
    from sklearn.metrics import adjusted_rand_score

    clst = TimeSeriesKMeans(
        averaging_method="mean",
        metric="dtw",
        distance_params={"window": 0.2},
        n_clusters=len(set(train_Y)),
        random_state=1,
        verbose=True,
    )
    clst.fit(trainX)
    preds = clst.predict(trainY)
    score = adjusted_rand_score(trainY, preds)
    print("Score = ", score)


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    numba.set_num_threads(1)
    torch.set_num_threads(1)

    clusterer = "kmeans"
    chris_config = True  # This is so chris doesn't have to change config each time
    tune = False
    normalise = True
    if sys.argv.__len__() > 1:  # cluster run, this is fragile, requires all args atm
        data_dir = sys.argv[1]
        results_dir = sys.argv[2]
        distance = sys.argv[3]
        dataset = sys.argv[4]
        # ADA starts indexing its jobs at 1, so we need to subtract 1
        resample = int(args[5]) - 1
        clusterer = sys.argv[6]
        if len(args) > 7:
            train_fold = args[7].lower() == "true"
        else:
            train_fold = False
        if len(args) > 8:
            averaging = args[8]
        else:
            averaging = "mean"
        if len(args) > 9:
            normalise = args[9].lower() == "true"
        else:
            normalise = False
        if averaging == "dba":
            results_dir = results_dir + clusterer + "_dba"
        if results_present(results_dir, clusterer, dataset, resample):
            print("Ignoring, results already present")


    elif chris_config is True:
        path = "C:/Users/chris/Documents/Masters"
        data_dir = os.path.abspath(f"{path}/datasets/Multivariate_ts/")
        results_dir = os.path.abspath(f"{path}/results/")
        dataset = "Handwriting"
        resample = 2
        averaging = "mean"
        train_fold = True
        distance = "dtw"

    else:  # Local run
        print(" Local Run")
        dataset = "Chinatown"
        data_dir = f"c:/temp/"
        results_dir = "./temp"
        resample = 0
        averaging = "dba"
        train_fold = True
        distance = "dtw"

    if isinstance(dataset, str):
        train_X, train_Y = load_ts(
            f"{data_dir}/{dataset}/{dataset}_TRAIN.ts", return_data_type="numpy2d"
        )
        test_X, test_Y = load_ts(
            f"{data_dir}/{dataset}/{dataset}_TEST.ts", return_data_type="numpy2d"
        )
    else:
        train_X, train_Y = dataset("train", return_X_y=True)
        test_X, test_Y = dataset("test", return_X_y=True)
    #    train_X = np.concatenate((train_X, test_X), axis=0)
    #    train_Y = np.concatenate((train_Y, test_Y), axis=0)
    #    _recreate_results(train_X, train_Y)
    #    import sys

    from sklearn.preprocessing import StandardScaler
    if normalise:
        s = StandardScaler()
        train_X = s.fit_transform(train_X.T)
        train_X = train_X.T
        test_X = s.fit_transform(test_X.T)
        test_X = test_X.T
    w = 1.0
    if tune:
        w = tune_window(distance, train_X, len(set(train_Y)))
        name = clusterer + "-" + distance + "-tuned"
    else:
        name = clusterer + "-" + distance
    w = 1.0
    if (distance == "wdtw" or distance == "dwdtw" or distance == "dtw" or distance ==
    "wdtw"):
        w = 0.2
    parameters = {
        "window": w,
        "epsilon": 0.05,
        "g": 0.05,
        "c": 1,
        "nu": 0.05,
        "lmbda": 1.0,
        "strategy": "independent",
    }
    average_params = {
        "averaging_distance_metric": distance,
        "medoids_distance_metric": distance,
    }
    if clusterer == "kmeans":
        print("running kmeans")
        format_kwargs = {**average_params, **parameters}
        clst = TimeSeriesKMeans(
            averaging_method=averaging,
            average_params=format_kwargs,
            metric=distance,
            distance_params=parameters,
            n_clusters=len(set(train_Y)),
            random_state=resample + 1,
            verbose=True,
        )
    else:
        clst = TimeSeriesKMedoids(
            metric=distance,
            distance_params=parameters,
            n_clusters=len(set(train_Y)),
            random_state=resample + 1,
        )

    run_clustering_experiment(
        train_X,
        clst,
        results_path=results_dir,
        trainY=train_Y,
        testX=test_X,
        testY=test_Y,
        cls_name=name,
        dataset_name=dataset,
        resample_id=resample,
        overwrite=False,
    )
    print("done")
