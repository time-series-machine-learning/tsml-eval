# -*- coding: utf-8 -*-
"""Tests for building HIVE-COTE from file."""

__author__ = ["ander-hg"]

import numpy as np


def ranklist(A):
    """Function to rank elements of list

    Parameters
    ----------
    A : list
        The accuracies.

    Returns
    -------
    R : list
        The ranks.
    """
    R = [0 for x in range(len(A))]

    # Counts the number of less than and equal elements of each element in A
    for i in range(len(A)):
        (less, equal) = (1, 1)
        for j in range(len(A)):
            if j != i and A[j] < A[i]:
                less += 1
            if j != i and A[j] == A[i]:
                equal += 1

        # The rank is the number of less than plus the midpoint of the number of ties
        R[i] = less + (equal - 1) / 2

    return R


def eval_mean_acc(datasets_names):
    """

    Parameters
    ----------
    datasets_names

    Returns
    -------

    """
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
        accuracies_array.append(folder_acc)

    # print(np.mean(accuracies_array, axis=1))
    print({file_paths[i]: accuracies_array[i] for i in range(len(file_paths))})
    # np.concatenate((a, b.T), axis=1)

    drop_no_tune = np.array(accuracies_array[0]) - np.array(accuracies_array[1])
    drop_non_strat = np.array(accuracies_array[0]) - np.array(accuracies_array[2])
    drop_strat = np.array(accuracies_array[0]) - np.array(accuracies_array[3])
    print("Accuracy drop > 1% on no_tune:")
    print(np.array(datasets_names)[drop_no_tune > 0.01])
    print("Accuracy drop > 1% on non_stratified:")
    print(np.array(datasets_names)[drop_non_strat > 0.01])
    print("Accuracy drop > 1% on stratified:")
    print(np.array(datasets_names)[drop_strat > 0.01])

    for acc_ds in accuracies_array:
        print(acc_ds)


def eval_alpha(datasets_names):
    """ALPHA VALUE OCCURRENCES

    Parameters
    ----------
    datasets_names

    Returns
    -------

    """
    alpha_values = []

    for dataset in datasets_names:
        dataset_alpha_values = []
        for resample in range(0, 30):
            f = open(
                "C:/Users/Ander/Documents/test_files/Matt/HC2-Tuned/Predictions/" + dataset + f"/testResample{resample}.csv",
                "r")
            lines = f.readlines()
            dataset_alpha_values.append(int(lines[1].split(",")[0].split(":")[1]))
        alpha_values.append(dataset_alpha_values)

    print("Alpha Values: ")
    print(alpha_values)

    c, d = np.unique(alpha_values, return_counts=True)
    dict_2 = {}
    for C, D in zip(c, d):
        dict_2[C] = D

    print("Number of occurrences of each alpha value:")
    print(dict_2)


def eval_type(path, datasets_names):
    """

    Parameters
    ----------
    path
    datasets_names

    Returns
    -------

    """
    f = open(path, "r")
    file = f.readlines()

    data_summary_labels = file[0].replace("\n", "").split(",")
    data_summary = []
    for line in file[1:]:
        list = line.replace("\n", "").split(",")
        data_summary.append(list[0:])

    data_type = [row[5] for row in data_summary]
    types, type_count = np.unique(data_type, return_counts=True)
    count_type_dict = {}
    index_type_dict = {}
    for A, B in zip(types, type_count):
        count_type_dict[A] = B
        index_type_dict[A] = [i for i, val in enumerate(data_type) if val == A]

    print("Number of occurrences of each type:")
    print(count_type_dict)
    print("Indexes of occurrences of each type:")
    print(index_type_dict)

    '''
    accuracies_list = []
    for dataset in datasets_names:
        dataset_acc = []
        for resample in range(0, 30):
            f = open( "C:/Users/Ander/Documents/test_files/Matt/HC2-Tuned/Predictions/" + dataset + f"/testResample{resample}.csv", "r")
            lines = f.readlines()
            dataset_acc.append(float(lines[2].split(",")[0]))
        accuracies_list.append(np.mean(dataset_acc))
    '''

    ''' ACC PER COMPONENT TESTS OF EACH TYPE '''

    components = [
        "Arsenal",
        "DrCIF-500",
        "STC-2Hour",
        "TDE",
    ]

    acc_per_components = []
    ranks_per_components = []
    for dataset in datasets_names:
        dataset_acc = []
        for path in components:
            component_acc = []
            for resample in range(0, 30):
                f = open(
                    "C:/Users/Ander/Documents/test_files/" + path + "/Predictions/" + dataset + f"/testResample{resample}.csv",
                    "r")
                lines = f.readlines()
                component_acc.append(float(lines[2].split(",")[0]))
            dataset_acc.append(np.mean(component_acc))
        ranks_per_components.append(ranklist(dataset_acc))
        acc_per_components.append(dataset_acc)

    print("Accuracies per component list: ")
    print(acc_per_components)
    print("Ranks per component list: ")
    print(ranks_per_components)

    for i, j in enumerate(components):
        a, b = np.unique([row[i] for row in ranks_per_components], return_counts=True)
        ranks_count_dict = {}
        for A, B in zip(a, b):
            ranks_count_dict[A] = B

        print(f"Rank of {j}:")
        print(ranks_count_dict)

    acc_comp_type = {}
    for t in types:
        acc_comp_type[t] = np.mean(np.array(acc_per_components)[index_type_dict[t]], axis=0)

    print("Accuracies of each component on each type: ")
    print("---------------------------------------------------------")
    print("-----------", components)
    for temp in acc_comp_type:
        print("{:<11} {}".format(temp, acc_comp_type[temp]))


if __name__ == "__main__":
    """
    """

    # reads the names of the datasets
    # "C:/Users/zrc22qwu/PycharmProjects/tsml-eval/tsml_eval/_wip/estimator_from_file/tests/test_files/UnivariateDatasets.txt",
    f = open(
        "C:/Users/Ander/Documents/test_files/UnivariateDatasets.txt",
        "r",
    )
    file = f.readlines()

    datasets_names = []
    for line in file:
        datasets_names.append(line.replace("\n", ""))

    # eval_mean_acc(datasets_names)

    eval_alpha(datasets_names)

    eval_type("C:/Users/Ander/Documents/test_files/DataSummary112.csv", datasets_names)
