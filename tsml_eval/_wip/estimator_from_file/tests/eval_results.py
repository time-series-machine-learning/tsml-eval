# -*- coding: utf-8 -*-
"""Tests for building HIVE-COTE from file."""

__author__ = ["ander-hg"]

import numpy as np
from sktime.datasets import load_arrow_head, load_italy_power_demand
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal
from sktime.utils.estimator_checks import check_estimator

from tsml_eval._wip.estimator_from_file.hivecote import FromFileHIVECOTE

def eval_hivecote_from_file():

    f = open(
        "C:/Users/zrc22qwu/PycharmProjects/tsml-eval/tsml_eval/_wip/estimator_from_file/tests/test_files/UnivariateDatasets.txt",
        #"C:/Users/Ander/git/tsml-estimator-evaluation/tsml_eval/_wip/estimator_from_file/tests/test_files/UnivariateDatasets.txt",
        "r",
    )
    file = f.readlines()

    datasets_names = []
    for line in file:
        datasets_names.append(line.replace("\n", ""))

    file_paths = [
        "HC2",
        "no_tune",
        "non_stratified",
        "stratified",
    ]

    accuracies_array = []
    for path in file_paths:
        folder_acc = []
        for dataset in datasets_names:
            dataset_acc = []
            for resample in range(0, 30):
                f = open("test_files/results/" + path + "/" + dataset + f"/testResample{resample}.csv", "r")
                lines = f.readlines()
                dataset_acc.append(float(lines[2].split(",")[0]))
            folder_acc.append(np.mean(dataset_acc))
        accuracies_array.append(np.mean(folder_acc))

    # print(np.mean(accuracies_array, axis=1))
    print({file_paths[i]: accuracies_array[i] for i in range(len(file_paths))})
    # np.concatenate((a, b.T), axis=1)


if __name__ == "__main__":
    """
    
    """
    eval_hivecote_from_file()