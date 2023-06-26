# -*- coding: utf-8 -*-
import os
import sys

import numpy as np

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import aeon.datasets.tsc_dataset_names as dataset_lists
import sklearn.metrics
from aeon.benchmarking.experiments import run_clustering_experiment
from aeon.clustering.k_means import TimeSeriesKMeans
from aeon.clustering.k_medoids import TimeSeriesKMedoids
from aeon.datasets import load_from_tsfile as load_ts
from aeon.datasets import load_gunpoint
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize


def test_experiment():
    print("starting test experiment")
    path = "C:/Users/chris/Documents/Masters"
    data_dir = os.path.abspath(f"{path}/datasets/Multivariate_ts/")
    results_dir = os.path.abspath(f"{path}/results/")
    dataset = "Handwriting"
    resample = 2
    averaging = "dba"
    tf = True
    parameters = {
        "window": 1.0,
        "epsilon": 0.05,
        "g": 0.05,
        "c": 1,
        "nu": 0.05,
        "lmbda": 1.0,
        "strategy": "independent",
    }
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
    from sklearn.preprocessing import StandardScaler

    s = StandardScaler()
    train_X = s.fit_transform(train_X.T)
    train_X = train_X.T
    test_X = s.fit_transform(test_X.T)
    test_X = test_X.T
    print("read datasets")

    train_X = train_X[:20]
    train_Y = train_Y[:20]
    test_X = test_X[:20]
    test_Y = test_Y[:20]
    # lcss bugs out, edr bugs out,

    for distance in ["msm", "dtw", "ddtw", "wdtw", "wddtw", "erp", "twe", "msm"]:
        print("running for distance: ", distance)
        average_params = {
            "averaging_distance_metric": distance,
            "medoids_distance_metric": distance,
            "strategy": "independent",
        }

        format_kwargs = {**average_params, **parameters}
        clst = TimeSeriesKMeans(
            averaging_method=averaging,
            average_params=format_kwargs,
            n_init=2,
            max_iter=30,
            metric=distance,
            distance_params=parameters,
            n_clusters=len(set(train_Y)),
            random_state=resample + 1,
            verbose=True,
        )
        print("----> Starting fit for:", distance)
        clst.fit(train_X)
        print("----> Finished fit")
        print("----> Starting predict for:", distance)
        clst.predict(test_X)
        print("----> Finished predict")
