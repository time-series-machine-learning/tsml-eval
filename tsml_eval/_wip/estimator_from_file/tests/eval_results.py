# -*- coding: utf-8 -*-
"""Tests for building HIVE-COTE from file."""

__author__ = ["ander-hg"]

import numpy as np
from sktime.datasets import load_arrow_head, load_italy_power_demand
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal
from sktime.utils.estimator_checks import check_estimator

from tsml_eval._wip.estimator_from_file.hivecote import FromFileHIVECOTE

def eval_hivecote_from_file():

    try:
        f = open(
            "C:/Users/Ander/git/tsml-estimator-evaluation/tsml_eval/_wip/estimator_from_file/tests/test_files/UnivariateDatasets.txt",
            "r",
        )
        file = f.readlines()
    finally:
        f.close()

    datasets_names = []
    for line in file:
        datasets_names.append(line.replace("\n", ""))

    file_paths = [
        "test_files/results/HC2/",
        "test_files/results/no_tune/",
        "test_files/results/non_stratified/",
        "test_files/results/stratified/",
    ]

    accuracies_array = []
    for path in file_paths:
        folder_acc = []
        for dataset in datasets_names:
            dataset_acc = []
            for resample in range(0, 30):
                try:
                    f = open(path + dataset + f"/trainResample{resample}.csv", "r")
                    lines = f.readlines()
                    dataset_acc.append(lines[2].split(",")[0])
                finally:
                    f.close()
            folder_acc.append(dataset_acc)
        accuracies_array.append(folder_acc)

    print(np.mean(accuracies_array, axis=1))

    # np.concatenate((a, b.T), axis=1)