# -*- coding: utf-8 -*-
"""Tests for building HIVE-COTE from file."""

__author__ = ["ander-hg"]

import numpy as np
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
import pandas as pd
import aeon
from aeon.datasets import load_from_tsfile_to_dataframe


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


def eval_mean_acc(caminho, datasets_names):
    """

    Parameters
    ----------
    datasets_names

    Returns
    -------

    """
    file_paths = [
        "HC2",
        "removing_worst",
        #"HC2-FF",
        #"HC2-tuned",
    ]

    accuracies_array = []
    for path in file_paths:
        folder_acc = []
        for dataset in datasets_names:
            dataset_acc = []
            for resample in range(0, 30):
                f = open(caminho + "test_files/results/" + path + "/Predictions/" + dataset + f"/testResample{resample}.csv", "r")
                lines = f.readlines()
                dataset_acc.append(float(lines[2].split(",")[0]))
            folder_acc.append(np.mean(dataset_acc))
        accuracies_array.append(folder_acc)

    # print(np.mean(accuracies_array, axis=1))
    #print({file_paths[i]: accuracies_array[i] for i in range(len(file_paths))})
    # np.concatenate((a, b.T), axis=1)



    components = [
        "Arsenal",
        "DrCIF-500",
        "STC-2Hour",
        "TDE",
        # "HC2",
    ]
    acc_per_components = []
    ranks_per_components = []
    for dataset in datasets_names:
        dataset_acc = []
        for compo in components:
            component_acc = []
            for resample in range(0, 30):
                f = open(
                    "C:/Users/zrc22qwu/Documents/test_files/" + compo + "/Predictions/" + dataset + f"/testResample{resample}.csv",
                    "r")
                lines = f.readlines()
                component_acc.append(float(lines[2].split(",")[0]))
            dataset_acc.append(np.mean(component_acc))
        ranks_per_components.append(ranklist(dataset_acc))
        acc_per_components.append(dataset_acc)
    print(acc_per_components)
    #variance = []
    min_div_max = []  # min/max acc
    for i, temp in enumerate(acc_per_components):
        #variance.append(np.var(temp))
        min_div_max.append(np.min(temp)/np.max(temp))
        temp2 = np.min(temp)
    temp = (np.array(accuracies_array[0]) - np.array(accuracies_array[1])) > 0.01
    print("Accuracy drop > 1% on remove worst:")
    keys = np.array(datasets_names)[temp]
    vals = [np.array(acc_per_components)[temp], np.array(min_div_max)[temp], np.array(accuracies_array[0])[temp], np.array(accuracies_array[1])[temp]]
    dict_acc = dict(zip(keys, zip(*vals)))
    for x in dict_acc:
        print(x, dict_acc[x])
    #print ("mean variance: ")
    #print(np.mean(np.array(variance)[temp]))
    print("mean min/max acc: ")
    print(np.mean(np.array(min_div_max)[temp]))

    temp = (np.array(accuracies_array[0]) - np.array(accuracies_array[1])) < -0.01
    print("Accuracy better by > 1% on remove_worst:")
    keys = np.array(datasets_names)[temp]
    vals = [np.array(acc_per_components)[temp], np.array(min_div_max)[temp], np.array(accuracies_array[0])[temp],
            np.array(accuracies_array[1])[temp]]
    dict_acc = dict(zip(keys, zip(*vals)))
    for x in dict_acc:
        print(x, dict_acc[x])
    #print("mean variance: ")
    #print(np.mean(np.array(variance)[temp]))
    print("mean min/max acc: ")
    print(np.mean(np.array(min_div_max)[temp]))


    print("All:")
    keys = np.array(datasets_names)
    vals = [np.array(acc_per_components), np.array(min_div_max),
            np.array(accuracies_array[0]),
            np.array(accuracies_array[1]),
            ]
    dict_acc = dict(zip(keys, zip(*vals)))
    '''
    for x in dict_acc:
        print(x, dict_acc[x])
    '''
    print("min_div_max < 0.7:")
    print(np.array(datasets_names)[np.array(min_div_max)<0.7])
    print("array(Arsenal, DrCIF, STC, TDE), min_div_max, acc_original, acc_RemoveWorst")
    for i in np.array(datasets_names)[np.array(min_div_max)<0.7]:
        print(dict_acc[i])

    print("min_div_max < 0.8:")
    print(np.array(datasets_names)[np.array(min_div_max) < 0.8])
    print("array(Arsenal, DrCIF, STC, TDE), min_div_max, acc_original, acc_RemoveWorst")
    for i in np.array(datasets_names)[np.array(min_div_max) < 0.8]:
        print(dict_acc[i])
    print("min_div_max < 0.9:")
    print(np.array(datasets_names)[np.array(min_div_max) < 0.9])
    print("array(Arsenal, DrCIF, STC, TDE), min_div_max, acc_original, acc_RemoveWorst")
    for i in np.array(datasets_names)[np.array(min_div_max) < 0.9]:
        print(dict_acc[i])
    #print("mean variance: ")
    #print(np.mean(np.array(variance)[temp]))
    #print(np.array(variance)[temp])
    #print("mean max_minus_min: ")
    #print(np.array(max_minus_min)[temp])

    #print("Accuracy drop > 1% on tuned_stratified:")
    #print(np.array(datasets_names)[drop_tuned > 0.01])
    '''
    for acc_ds in accuracies_array:
        print(acc_ds)
        print(np.mean(acc_ds))
    '''

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
        #"HC2",
    ]
    '''
    f = open(
        "C:/Users/zrc22qwu/PycharmProjects/tsml-eval/tsml_eval/_wip/estimator_from_file/tests/results/rank_comp.txt", # _x_hc
        "a")
    f.write(f"Dataset_name ")
    for temp in components:
        f.write(f"& {temp} ")
        f.write("\\\\")
    f.close()
    '''
    acc_per_components = []
    ranks_per_components = []
    for dataset in datasets_names:
        dataset_acc = []
        for path in components:
            component_acc = []
            for resample in range(0, 30):
                f = open(
                    "C:/Users/zrc22qwu/Documents/test_files/" + path + "/Predictions/" + dataset + f"/testResample{resample}.csv",
                    "r")
                lines = f.readlines()
                component_acc.append(float(lines[2].split(",")[0]))
            dataset_acc.append(np.mean(component_acc))
        ranks_per_components.append(ranklist(dataset_acc))
        acc_per_components.append(dataset_acc)
        '''
        f = open(
            "C:/Users/zrc22qwu/PycharmProjects/tsml-eval/tsml_eval/_wip/estimator_from_file/tests/results/rank_comp.txt",
            "a")
        f.write(f"\n{dataset} ")
        for temp in dataset_acc:
            f.write(f"& {temp:.4f} ")
        f.write("\\\\")
        f.close()
        '''

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


def feature_extract():
    # extracted_features = extract_features(timeseries, column_id="id", column_sort="time")

    #data_name = 'ACSF1'
    data_name = 'synthetic_control-mld.csv'

    print("Starting the preprocessing of " + data_name + " dataset with TSFRESH")

    '''
    Training dataset
    '''
    print("Loading training dataset")

    file_path = "C:/Users/zrc22qwu/PycharmProjects/metalcats-main/metalcats-main/data/" + data_name + "/" + data_name + "_TRAIN.ts"
    #data_X, data_y = load_from_tsfile_to_dataframe(file_path)

    #teste = []

    #for i, ts in enumerate(data_X):
    #    for obs in ts:
    #        teste.append([i, obs, data_y[0]])

    file_path = "C:/Users/zrc22qwu/PycharmProjects/metalcats-main/metalcats-main/data/" + data_name
    f = open(file_path)
    linhas = f.readlines()

    data = []
    for linha in linhas:
        data.append(linha)

    teste = []

    for i, ts in enumerate(data):
        print(ts)
        for obs in [float(x) for x in ts.split(",")]:
            teste.append([i, obs, int((i + 99) / 100)])

    print("Done")
    print("Extracting features")

    from sklearn.model_selection import train_test_split

    cols = ["id", "x", "y"]
    df = pd.DataFrame(teste, columns=cols)


    X_train, X_test = train_test_split(df, test_size=.3, stratify=df['y'])

    X_train_tsfresh = extract_features(X_train, default_fc_parameters=MinimalFCParameters(), column_id='id')
    X_train_tsfresh.to_csv("C:/Users/zrc22qwu/PycharmProjects/metalcats-main/metalcats-main/tsfresh/" + data_name + "_TRAIN.csv")
    X_test_tsfresh = extract_features(X_test, default_fc_parameters=MinimalFCParameters(), column_id='id')
    X_test_tsfresh.to_csv("C:/Users/zrc22qwu/PycharmProjects/metalcats-main/metalcats-main/tsfresh/" + data_name + "_TEST.csv")
    print("Done!\n")

if __name__ == "__main__":
    """
    """

    # reads the names of the datasets
    # "C:/Users/zrc22qwu/PycharmProjects/tsml-eval/tsml_eval/_wip/estimator_from_file/tests/test_files/UnivariateDatasets.txt",
    f = open(
        "C:/Users/zrc22qwu/Documents/UnivariateDatasets.txt",
        "r",
    )
    file = f.readlines()

    datasets_names = []
    for line in file:
        datasets_names.append(line.replace("\n", ""))

    path = "C:/Users/zrc22qwu/Documents/"
    eval_mean_acc(path, datasets_names)

    #eval_alpha(datasets_names)

    #f = open("C:/Users/zrc22qwu/PycharmProjects/tsml-eval/tsml_eval/_wip/estimator_from_file/tests/results/rank_comp.txt", "w")
    #f.close()

    #eval_type("C:/Users/zrc22qwu/Documents/DataSummary112.csv", datasets_names)

    #feature_extract()
