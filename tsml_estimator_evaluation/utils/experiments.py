# -*- coding: utf-8 -*-

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os

from sktime.datasets import load_from_tsfile_to_dataframe as load_ts
from sktime.datasets import tsc_dataset_names


def demo_loading():
    """Test to check dataset loading of univariate and multivariate problems."""
    for i in range(0, len(tsc_dataset_names.univariate)):
        data_dir = "../../"
        dataset = tsc_dataset_names.univariate[i]
        trainX, trainY = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
        testX, testY = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
        print("Loaded " + dataset + " in position " + str(i))
        print("Train X shape :")
        print(trainX.shape)
        print("Train Y shape :")
        print(trainY.shape)
        print("Test X shape :")
        print(testX.shape)
        print("Test Y shape :")
        print(testY.shape)
    for i in range(16, len(tsc_dataset_names.multivariate)):
        data_dir = "E:/mtsc_ts/"
        dataset = tsc_dataset_names.multivariate[i]
        print("Loading " + dataset + " in position " + str(i) + ".......")
        trainX, trainY = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
        testX, testY = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
        print("Loaded " + dataset)
        print("Train X shape :")
        print(trainX.shape)
        print("Train Y shape :")
        print(trainY.shape)
        print("Test X shape :")
        print(testX.shape)
        print("Test Y shape :")
        print(testY.shape)


def results_present(results_path, cls_name, dataset, resample_id):
    full_path = (
        results_path
        + "/"
        + cls_name
        + "/Predictions/"
        + dataset
        + "/testResample"
        + str(resample_id)
        + ".csv"
    )
    full_path2 = (
        results_path
        + "/"
        + cls_name
        + "/Predictions/"
        + dataset
        + "/trainResample"
        + str(resample_id)
        + ".csv"
    )
    if os.path.exists(full_path) and os.path.exists(full_path2):
        return True
    return False
