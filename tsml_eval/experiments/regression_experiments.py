# -*- coding: utf-8 -*-
"""Regressor Experiments: code to run experiments and generate results file in
standard format.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format. It is cloned from
classification_experiments, we should condense it all to one.
"""

__author__ = ["TonyBagnall"]

import os

# Remove if not running on cluster?
os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import time
from datetime import datetime

import numba
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sktime.datasets import load_from_tsfile_to_dataframe as load_ts
from sktime.datasets import write_results_to_uea_format

from tsml_eval.experiments.set_regressor import set_regressor
from tsml_eval.utils.experiments import results_present


def resample(train_X, train_y, test_X, test_y, random_state):
    """Resample data without replacement using a random state.

    Reproducable resampling. Combines train and test, resamples to get the same class
    distribution, then returns new train and test.

    Parameters
    ----------
    train_X : pd.DataFrame
        train data attributes in sktime pandas format.
    train_y : np.array
        train data class labels.
    test_X : pd.DataFrame
        test data attributes in sktime pandas format.
    test_y : np.array
        test data class labels as np array.
    random_state : int
        seed to enable reproducable resamples

    Returns
    -------
    new train and test attributes and class labels.
    """
    all_targets = np.concatenate((train_y, test_y), axis=None)
    all_data = pd.concat([train_X, test_X])
    all_data["target"] = all_targets

    shuffled = all_data.sample(frac=1, random_state=random_state)

    # extract and remove the target column
    all_targets = shuffled["target"].to_numpy()
    shuffled = shuffled.drop("target", axis=1)

    # split the shuffled data into train and test
    train_cases = train_y.size
    train_X = shuffled.iloc[:train_cases]
    test_X = shuffled.iloc[train_cases:]
    train_y = all_targets[:train_cases]
    test_y = all_targets[train_cases:]

    # reset indexes to conform to sktime format.
    train_X = train_X.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)

    return train_X, train_y, test_X, test_y


def run_regression_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    regressor,
    results_path,
    regressor_name="",
    dataset="",
    resample_id=0,
    train_file=False,
    test_file=True,
):
    """Run a regression experiment and save the results to file.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv.

    Parameters
    ----------
    X_train : pd.DataFrame or np.array
        The data to train the classifier.
    y_train : np.array, default = None
        Training data class labels.
    X_test : pd.DataFrame or np.array, default = None
        The data used to test the trained classifier.
    y_test : np.array, default = None
        Testing data class labels.
    regressor : BaseRegressor
        Regressor to be used in the experiment.
    results_path : str
        Location of where to write results. Any required directories will be created.
    regressor_name : str, default=""
        Name of the Regressor to use in file writing.
    dataset : str, default=""
        Name of problem to use in file writing.
    resample_id : int, default=0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    train_file : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the regressor can produce its
        own estimates, those are used instead.
    test_file : bool, default=True:
         Whether to generate test files or not. If the regressor can generate its own
         train probabilities, the classifier will be built but no file will be output.
    """
    if not test_file and not train_file:
        raise Exception(
            "Both test_file and train_file are set to False. "
            "At least one must be output."
        )

    regressor_train_preds = train_file and callable(
        getattr(regressor, "_get_train_preds", None)
    )
    build_time = -1

    if test_file or regressor_train_preds:
        start = int(round(time.time() * 1000))
        regressor.fit(X_train, y_train)
        build_time = int(round(time.time() * 1000)) - start

    if test_file:
        start = int(round(time.time() * 1000))
        preds = regressor.predict(X_test)
        test_time = int(round(time.time() * 1000)) - start

        if "composite" in regressor_name.lower():
            second = "Para info too long!"
        else:
            second = str(regressor.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")

        mse = mean_squared_error(y_test, preds)

        third = f"{mse},{build_time},{test_time},-1,-1,,-1,-1"

        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            first_line_comment="Generated by regression_experiments.py on "
            + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            timing_type="MILLISECONDS",
            output_path=results_path,
            estimator_name=regressor_name,
            resample_seed=resample_id,
            y_pred=preds,
            dataset_name=dataset,
            y_true=y_test,
            split="TEST",
            full_path=False,
        )

    if train_file:
        start = int(round(time.time() * 1000))
        if regressor_train_preds:  # Normally can only do this if test has been built
            train_preds = regressor._get_train_preds(X_train, y_train)
        else:
            cv_size = min(10, len(y_train))
            train_preds = cross_val_predict(regressor, X_train, y=y_train, cv=cv_size)
        train_time = int(round(time.time() * 1000)) - start

        if "composite" in regressor_name.lower():
            second = "Para info too long!"
        else:
            second = str(regressor.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")

        mse = mean_squared_error(y_train, train_preds)

        third = f"{mse},{build_time},-1,-1,-1,,{train_time},{build_time + train_time}"

        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            first_line_comment="Generated by regression_experiments.py on "
            + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            timing_type="MILLISECONDS",
            output_path=results_path,
            estimator_name=regressor_name,
            resample_seed=resample_id,
            y_pred=train_preds,
            dataset_name=dataset,
            y_true=y_train,
            split="TRAIN",
            full_path=False,
        )


def load_and_run_regression_experiment(
    problem_path,
    results_path,
    dataset,
    regressor,
    resample_id=0,
    regressor_name=None,
    overwrite=False,
    build_train=False,
    predefined_resample=False,
):
    """Load a dataset and run a classification experiment.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv.

    Parameters
    ----------
    problem_path : str
        Location of problem files, full path.
    results_path : str
        Location of where to write results. Any required directories will be created.
    dataset : str
        Name of problem. Files must be  <problem_path>/<dataset>/<dataset>+"_TRAIN.ts",
        same for "_TEST".
    regressor : BaseClassifier
        Classifier to be used in the experiment, if none is provided one is selected
        using cls_name using resample_id as a seed.
    regressor_name : str, default = None
        Name of classifier used in writing results. If none the name is taken from
        the classifier
    resample_id : int, default=0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    overwrite : bool, default=False
        If set to False, this will only build results if there is not a result file
        already present. If True, it will overwrite anything already there.
    build_train : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the classifier can produce its
        own estimates, those are used instead.
    predefined_resample : bool, default=False
        Read a predefined resample from file instead of performing a resample. If True
        the file format must include the resample_id at the end of the dataset name i.e.
        <problem_path>/<dataset>/<dataset>+<resample_id>+"_TRAIN.ts".
    """
    if regressor_name is None:
        regressor_name = type(regressor).__name__

    # Check which files exist, if both exist, exit
    build_test = True
    if not overwrite:
        full_path = (
            results_path
            + "/"
            + regressor_name
            + "/Predictions/"
            + dataset
            + "/testResample"
            + str(resample_id)
            + ".csv"
        )

        if os.path.exists(full_path):
            build_test = False

        if build_train:
            full_path = (
                results_path
                + "/"
                + regressor_name
                + "/Predictions/"
                + dataset
                + "/trainResample"
                + str(resample_id)
                + ".csv"
            )

            if os.path.exists(full_path):
                build_train = False

        if build_train is False and not build_test:
            return

    if predefined_resample:
        X_train, y_train = load_ts(
            problem_path + dataset + "/" + dataset + str(resample_id) + "_TRAIN.ts"
        )
        X_test, y_test = load_ts(
            problem_path + dataset + "/" + dataset + str(resample_id) + "_TEST.ts"
        )
    else:
        X_train, y_train = load_ts(problem_path + dataset + "/" + dataset + "_TRAIN.ts")
        X_test, y_test = load_ts(problem_path + dataset + "/" + dataset + "_TEST.ts")
        if resample_id != 0:
            X_train, y_train, X_test, y_test = resample(
                X_train, y_train, X_test, y_test, resample_id
            )

    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    run_regression_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        regressor,
        results_path,
        regressor_name=regressor_name,
        dataset=dataset,
        resample_id=resample_id,
        train_file=build_train,
        test_file=build_test,
    )


def run_experiment(args, overwrite=False):
    numba.set_num_threads(1)
    torch.set_num_threads(1)

    # cluster run (with args), this is fragile
    if args.__len__() > 1:  # cluster run, this is fragile
        print("Input args = ", args)
        data_dir = args[1]
        results_dir = args[2]
        regressor_name = args[3]
        dataset = args[4]
        # ADA starts indexing its jobs at 1, so we need to subtract 1
        resample = int(args[5]) - 1

        if len(args) > 6:
            train_fold = args[6].lower() == "true"
        else:
            train_fold = False

        if len(args) > 7:
            predefined_resample = args[7].lower() == "true"
        else:
            predefined_resample = False

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if not overwrite and results_present(
            results_dir, regressor_name, dataset, resample
        ):
            print("Ignoring, results already present")
        else:
            load_and_run_regression_experiment(
                problem_path=data_dir,
                results_path=results_dir,
                regressor=set_regressor(regressor_name, resample, train_fold),
                regressor_name=regressor_name,
                dataset=dataset,
                resample_id=resample,
                build_train=train_fold,
                predefined_resample=predefined_resample,
                overwrite=overwrite,
            )
    # local run (no args)
    else:
        print(" Local Run of TimeSeriesForestRegressor")
        data_dir = "../../../time_series_regression/new_datasets/"
        results_dir = "../"
        regressor_name = "svr"
        dataset = "Covid3Months"
        resample = 0
        train_fold = False
        predefined_resample = False
        regressor = set_regressor(regressor_name)
        print(f"Local Run of {regressor.__class__.__name__}.")

        load_and_run_regression_experiment(
            problem_path=data_dir,
            results_path=results_dir,
            regressor=regressor,
            regressor_name=regressor_name,
            dataset=dataset,
            resample_id=resample,
            build_train=train_fold,
            predefined_resample=predefined_resample,
            overwrite=True,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
