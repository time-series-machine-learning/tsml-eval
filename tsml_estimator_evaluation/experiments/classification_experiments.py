# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

__author__ = ["TonyBagnall"]

import os
import sys

import sktime.datasets.tsc_dataset_names as dataset_lists
from set_classifier import set_classifier
from sktime.benchmarking.experiments import load_and_run_classification_experiment
from sktime.datasets import load_from_tsfile_to_dataframe as load_ts

# os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
# os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
# os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!


"""Prototype mechanism for testing classifiers on the UCR format. This mirrors the
mechanism used in Java,
https://github.com/TonyBagnall/uea-tsc/tree/master/src/main/java/experiments
but is not yet as engineered. However, if you generate results using the method
recommended here, they can be directly and automatically compared to the results
generated in java.
"""


from sktime.registry import all_estimators


def list_all_multivariate_capable_classifiers():
    """Return a list of all multivariate capable classifiers in sktime."""
    cls = []
    from sktime.registry import all_estimators

    cls = all_estimators(
        estimator_types="classifier", filter_tags={"capability:multivariate": True}
    )
    names = [i for i, _ in cls]
    return names


def list_estimators(
    estimator_type="classifier",
    multivariate_only=False,
    univariate_only=False,
    dictionary=True,
):
    """Return a list of all estimators of given type in sktime."""
    cls = []
    filter_tags = {}
    if multivariate_only:
        filter_tags["capability:multivariate"] = True
    if univariate_only:
        filter_tags["capability:multivariate"] = False
    cls = all_estimators(estimator_types=estimator_type, filter_tags=filter_tags)
    names = [i for i, _ in cls]
    return names


str=list_estimators(estimator_type="classifier")
print(str)
for s in str:
    print(f"\"{s}\",")


def demo_loading():
    """Test function to check dataset loading of univariate and multivaria problems."""
    for i in range(0, len(dataset_lists.univariate)):
        data_dir = "../../"
        dataset = dataset_lists.univariate[i]
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
    for i in range(16, len(dataset_lists.multivariate)):
        data_dir = "E:/mtsc_ts/"
        dataset = dataset_lists.multivariate[i]
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


def run_experiment(args):
    if args.__len__() > 1:  # cluster run, this is fragile
        print(" Input args = ", sys.argv)
        data_dir = sys.argv[1]
        results_dir = sys.argv[2]
        classifier = sys.argv[3]
        dataset = sys.argv[4]
        resample = int(sys.argv[5]) - 1

        if len(sys.argv) > 6:
            tf = sys.argv[6].lower() == "true"
        else:
            tf = False

        if len(sys.argv) > 7:
            predefined_resample = sys.argv[7].lower() == "true"
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
                classifier=set_classifier(classifier, resample, tf),
                cls_name=classifier,
                dataset=dataset,
                resample_id=resample,
                build_train=tf,
                predefined_resample=predefined_resample,
            )
    else:  # Local run
        print(" Local Run of HIVECOTE2 with threading")
        data_dir = "/home/ajb/Data/"
        results_dir = "/home/ajb/temp/"
        cls_name = "HIVECOTEV2"
        n_jobs = 14
        classifier = set_classifier(cls_name, n_jobs=n_jobs)
        dataset = "Blink"
        resample = 0
        tf = False
        predefined_resample = False

        load_and_run_classification_experiment(
            overwrite=False,
            problem_path=data_dir,
            results_path=results_dir,
            cls_name=cls_name,
            classifier=classifier,
            dataset=dataset,
            resample_id=resample,
            build_train=tf,
            predefined_resample=predefined_resample,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
