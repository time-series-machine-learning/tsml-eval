"""Utility functions for experiments."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

__all__ = [
    "resample_data",
    "stratified_resample_data",
    "write_classification_results",
    "write_regression_results",
    "write_clustering_results",
    "write_forecasting_results",
    "write_results_to_tsml_format",
    "fix_broken_second_line",
    "compare_result_file_resample",
    "assign_gpu",
    "timing_benchmark",
    "estimator_attributes_to_file",
]

import os
import time
from collections.abc import Sequence

import gpustat
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from tsml.datasets import load_from_ts_file

from tsml_eval.utils.validation import (
    _check_classification_third_line,
    _check_clustering_third_line,
    _check_regression_third_line,
)


def resample_data(X_train, y_train, X_test, y_test, random_state=None):
    """Resample data without replacement using a random state.

    Reproducible resampling. Combines train and test, randomly resamples, then returns
    new train and test.

    Parameters
    ----------
    X_train : np.ndarray or list of np.ndarray
        Train data in a 2d or 3d ndarray or list of arrays.
    y_train : np.ndarray
        Train data labels.
    X_test : np.ndarray or list of np.ndarray
        Test data in a 2d or 3d ndarray or list of arrays.
    y_test : np.ndarray
        Test data labels.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    train_X : np.ndarray or list of np.ndarray
        New train data.
    train_y : np.ndarray
        New train labels.
    test_X : np.ndarray or list of np.ndarray
        New test data.
    test_y : np.ndarray
        New test labels.
    """
    if isinstance(X_train, np.ndarray):
        is_array = True
    elif isinstance(X_train, list):
        is_array = False
    else:
        raise ValueError(
            "X_train must be a np.ndarray array or list of np.ndarray arrays"
        )

    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = (
        np.concatenate([X_train, X_test], axis=0) if is_array else X_train + X_test
    )

    # shuffle data indices
    rng = check_random_state(random_state)
    indices = np.arange(len(all_data), dtype=int)
    rng.shuffle(indices)

    train_cases = y_train.size
    train_indices = indices[:train_cases]
    test_indices = indices[train_cases:]

    # split the shuffled data into train and test
    X_train = (
        all_data[train_indices] if is_array else [all_data[i] for i in train_indices]
    )
    y_train = all_labels[train_indices]
    X_test = all_data[test_indices] if is_array else [all_data[i] for i in test_indices]
    y_test = all_labels[test_indices]

    return X_train, y_train, X_test, y_test


def load_experiment_data(
    problem_path: str,
    dataset: str,
    resample_id: int,
    predefined_resample: bool,
):
    """Load data for experiments.

    Parameters
    ----------
    problem_path : str
        Path to the problem folder.
    dataset : str
        Name of the dataset.
    resample_id : int or None
        Id of the data resample to use.
    predefined_resample : boolean
        If True, use the predefined resample.

    Returns
    -------
    X_train : np.ndarray or list of np.ndarray
        Train data in a 2d or 3d ndarray or list of arrays.
    y_train : np.ndarray
        Train data labels.
    X_test : np.ndarray or list of np.ndarray
        Test data in a 2d or 3d ndarray or list of arrays.
    y_test : np.ndarray
        Test data labels.
    resample : boolean
        If True, the data is to be resampled.
    """
    if resample_id is not None and predefined_resample:
        resample_str = "" if resample_id is None else str(resample_id)

        X_train, y_train = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TRAIN.ts"
        )
        X_test, y_test = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TEST.ts"
        )

        resample_data = False
    else:
        X_train, y_train = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}_TRAIN.ts"
        )
        X_test, y_test = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}_TEST.ts"
        )

        resample_data = True if resample_id != 0 else False

    return X_train, y_train, X_test, y_test, resample_data


def stratified_resample_data(X_train, y_train, X_test, y_test, random_state=None):
    """Stratified resample data without replacement using a random state.

    Reproducible resampling. Combines train and test, resamples to get the same class
    distribution, then returns new train and test.

    Parameters
    ----------
    X_train : np.ndarray or list of np.ndarray
        Train data in a 2d or 3d ndarray or list of arrays.
    y_train : np.ndarray
        Train data labels.
    X_test : np.ndarray or list of np.ndarray
        Test data in a 2d or 3d ndarray or list of arrays.
    y_test : np.ndarray
        Test data labels.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    train_X : np.ndarray or list of np.ndarray
        New train data.
    train_y : np.ndarray
        New train labels.
    test_X : np.ndarray or list of np.ndarray
        New test data.
    test_y : np.ndarray
        New test labels.
    """
    if isinstance(X_train, np.ndarray):
        is_array = True
    elif isinstance(X_train, list):
        is_array = False
    else:
        raise ValueError(
            "X_train must be a np.ndarray array or list of np.ndarray arrays"
        )

    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = (
        np.concatenate([X_train, X_test], axis=0) if is_array else X_train + X_test
    )

    # shuffle data indices
    rng = check_random_state(random_state)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    # ensure same classes exist in both train and test
    assert list(unique_train) == list(unique_test)

    if is_array:
        shape = list(X_train.shape)
        shape[0] = 0

    X_train = np.zeros(shape) if is_array else []
    y_train = np.zeros(0)
    X_test = np.zeros(shape) if is_array else []
    y_test = np.zeros(0)

    # for each class
    for label_index in range(len(unique_train)):
        # get the indices of all instances with this class label and shuffle them
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]
        rng.shuffle(indices)

        train_cases = counts_train[label_index]
        train_indices = indices[:train_cases]
        test_indices = indices[train_cases:]

        # extract data from corresponding indices
        train_cases = (
            all_data[train_indices]
            if is_array
            else [all_data[i] for i in train_indices]
        )
        train_labels = all_labels[train_indices]
        test_cases = (
            all_data[test_indices] if is_array else [all_data[i] for i in test_indices]
        )
        test_labels = all_labels[test_indices]

        # concat onto current data from previous loop iterations
        X_train = (
            np.concatenate([X_train, train_cases], axis=0)
            if is_array
            else X_train + train_cases
        )
        y_train = np.concatenate([y_train, train_labels], axis=None)
        X_test = (
            np.concatenate([X_test, test_cases], axis=0)
            if is_array
            else X_test + test_cases
        )
        y_test = np.concatenate([y_test, test_labels], axis=None)

    return X_train, y_train, X_test, y_test


def write_classification_results(
    predictions,
    probabilities,
    class_labels,
    classifier_name,
    dataset_name,
    file_path,
    full_path=True,
    split=None,
    resample_id=None,
    time_unit="N/A",
    first_line_comment=None,
    parameter_info="No Parameter Info",
    accuracy=-1,
    fit_time=-1,
    predict_time=-1,
    benchmark_time=-1,
    memory_usage=-1,
    n_classes=-1,
    train_estimate_method="",
    train_estimate_time=-1,
    fit_and_estimate_time=-1,
):
    """Write the predictions for a classification experiment in the format used by tsml.

    Parameters
    ----------
    predictions : np.array
        The predicted values to write to file. Must be the same length as labels.
    probabilities : np.ndarray
        Estimated class probabilities. These are written after the
        predicted values for each case.
    class_labels : np.array
        The actual class values written to file with the predicted values.
    classifier_name : str
        Name of the classifier that made the predictions. Written to file and can
        determine file structure if full_path is False.
    dataset_name : str
        Name of the problem the classifier was built on.
    file_path : str
        Path to write the results file to or the directory to build the default file
        structure if full_path is False.
    full_path : boolean, default=True
        If True, results are written directly to the directory passed in file_path.
        If False, then a standard file structure using the classifier and dataset names
        is created and used to write the results file.
    split : str or None, default=None
        Either None, 'TRAIN' or 'TEST'. Influences the result file name and first line
        of the file.
    resample_id : int or None, default=None
        Indicates what random seed was used to resample the data or used as a
        random_state for the classifier.
    time_unit : str, default="N/A"
        The format used for timings in the file, i.e. 'Seconds', 'Milliseconds',
        'Nanoseconds'
    first_line_comment : str or None, default=None
        Optional comment appended to the end of the first line, i.e. the file used to
        generate the results.
    parameter_info : str, default="No Parameter Info"
        Unstructured estimator dependant information, i.e. estimator parameters or
        values from the model build.
    accuracy: float, default=-1
        The accuracy of the predictions.
    fit_time : int, default=-1
        The time taken to fit the classifier.
    predict_time : int, default=-1
        The time taken to predict the class labels.
    benchmark_time : int, default=-1
        A benchmark time for the hardware used to scale other timings.
    memory_usage : int, default=-1
        The memory usage of the classifier.
    n_classes : int, default=-1
        The number of classes in the dataset.
    train_estimate_method : str, default=""
        The method used to generate predictions for results on training data.
    train_estimate_time : int, default=-1
        The time taken to generate predictions for results on training data.
    fit_and_estimate_time : int, default=-1
        The time taken to fit the classifier to build and generate predictions for
        results on training data.

        This is not necessarily always going to be fit_time + train_estimate_time,
        i.e. if an estimate requires the model to be fit, fit_time would be
        included in the train_estimate_time value. In this case fit_time +
        train_estimate_time would time fitting the model twice.
    """
    if len(predictions) != probabilities.shape[0] != len(class_labels):
        raise IndexError(
            "The number of predicted values is not the same as the number of actual "
            "class values."
        )

    if n_classes > -1 and n_classes != probabilities.shape[1]:
        raise IndexError(
            "The number of classes is not the same as the number of probability "
            "values for each case."
        )

    third_line = (
        f"{accuracy},"
        f"{fit_time},"
        f"{predict_time},"
        f"{benchmark_time},"
        f"{memory_usage},"
        f"{n_classes},"
        f"{train_estimate_method},"
        f"{train_estimate_time},"
        f"{fit_and_estimate_time}"
    )

    write_results_to_tsml_format(
        predictions,
        class_labels,
        classifier_name,
        dataset_name,
        file_path,
        predicted_probabilities=probabilities,
        full_path=full_path,
        split=split,
        resample_id=resample_id,
        time_unit=time_unit,
        first_line_comment=first_line_comment,
        second_line=parameter_info,
        third_line=third_line,
    )


def write_regression_results(
    predictions,
    labels,
    regressor_name,
    dataset_name,
    file_path,
    full_path=True,
    split=None,
    resample_id=None,
    time_unit="N/A",
    first_line_comment=None,
    parameter_info="No Parameter Info",
    mse=-1,
    fit_time=-1,
    predict_time=-1,
    benchmark_time=-1,
    memory_usage=-1,
    train_estimate_method="",
    train_estimate_time=-1,
    fit_and_estimate_time=-1,
):
    """Write the predictions for a regression experiment in the format used by tsml.

    Parameters
    ----------
    predictions : np.array
        The predicted values to write to file. Must be the same length as labels.
    labels : np.array
        The actual label values written to file with the predicted values.
    regressor_name : str
        Name of the regressor that made the predictions. Written to file and can
        determine file structure if full_path is False.
    dataset_name : str
        Name of the problem the regressor was built on.
    file_path : str
        Path to write the results file to or the directory to build the default file
        structure if full_path is False.
    full_path : boolean, default=True
        If True, results are written directly to the directory passed in file_path.
        If False, then a standard file structure using the regressor and dataset names
        is created and used to write the results file.
    split : str or None, default=None
        Either None, 'TRAIN' or 'TEST'. Influences the result file name and first line
        of the file.
    resample_id : int or None, default=None
        Indicates what random seed was used to resample the data or used as a
        random_state for the regressor.
    time_unit : str, default="N/A"
        The format used for timings in the file, i.e. 'Seconds', 'Milliseconds',
        'Nanoseconds'
    first_line_comment : str or None, default=None
        Optional comment appended to the end of the first line, i.e. the file used to
        generate the results.
    parameter_info : str, default="No Parameter Info"
        Unstructured estimator dependant information, i.e. estimator parameters or
        values from the model build.
    mse: float, default=-1
        The mean squared error of the predictions.
    fit_time : int, default=-1
        The time taken to fit the regressor.
    predict_time : int, default=-1
        The time taken to predict the regression labels.
    benchmark_time : int, default=-1
        A benchmark time for the hardware used to scale other timings.
    memory_usage : int, default=-1
        The memory usage of the regressor.
    train_estimate_method : str, default=""
        The method used to generate predictions for results on training data.
    train_estimate_time : int, default=-1
        The time taken to generate predictions for results on training data.
    fit_and_estimate_time : int, default=-1
        The time taken to fit the regressor to build and generate predictions for
        results on training data.

        This is not necessarily always going to be fit_time + train_estimate_time,
        i.e. if an estimate requires the model to be fit, fit_time would be
        included in the train_estimate_time value. In this case fit_time +
        train_estimate_time would time fitting the model twice.
    """
    third_line = (
        f"{mse},"
        f"{fit_time},"
        f"{predict_time},"
        f"{benchmark_time},"
        f"{memory_usage},"
        f"{train_estimate_method},"
        f"{train_estimate_time},"
        f"{fit_and_estimate_time}"
    )

    write_results_to_tsml_format(
        predictions,
        labels,
        regressor_name,
        dataset_name,
        file_path,
        full_path=full_path,
        split=split,
        resample_id=resample_id,
        time_unit=time_unit,
        first_line_comment=first_line_comment,
        second_line=parameter_info,
        third_line=third_line,
    )


def write_clustering_results(
    cluster_predictions,
    cluster_probabilities,
    class_labels,
    clusterer_name,
    dataset_name,
    file_path,
    full_path=True,
    split=None,
    resample_id=None,
    time_unit="N/A",
    first_line_comment=None,
    parameter_info="No Parameter Info",
    clustering_accuracy=-1,
    fit_time=-1,
    predict_time=-1,
    benchmark_time=-1,
    memory_usage=-1,
    n_classes=-1,
    n_clusters=-1,
):
    """Write the predictions for a clustering experiment in the format used by tsml.

    Parameters
    ----------
    cluster_predictions : np.array
        The predicted values to write to file. Must be the same length as labels.
    cluster_probabilities : np.ndarray
        Estimated cluster probabilities. These are written after the predicted values
        for each case.
    class_labels : np.array
        The actual class values written to file with the predicted values. If no label
        is available for a case, a NaN value should be substituted.
    clusterer_name : str
        Name of the clusterer that made the predictions. Written to file and can
        determine file structure if full_path is False.
    dataset_name : str
        Name of the problem the clusterer was built on.
    file_path : str
        Path to write the results file to or the directory to build the default file
        structure if full_path is False.
    full_path : boolean, default=True
        If True, results are written directly to the directory passed in file_path.
        If False, then a standard file structure using the clusterer and dataset names
        is created and used to write the results file.
    split : str or None, default=None
        Either None, 'TRAIN' or 'TEST'. Influences the result file name and first line
        of the file.
    resample_id : int or None, default=None
        Indicates what random seed was used to resample the data or used as a
        random_state for the clusterer.
    time_unit : str, default="N/A"
        The format used for timings in the file, i.e. 'Seconds', 'Milliseconds',
        'Nanoseconds'
    first_line_comment : str or None, default=None
        Optional comment appended to the end of the first line, i.e. the file used to
        generate the results or a dictionary linking label indices to actual values.
    parameter_info : str, default="No Parameter Info"
        Unstructured estimator dependant information, i.e. estimator parameters or
        values from the model build.
    clustering_accuracy : float, default=-1
        The clustering accuracy of the predictions.
    fit_time : int, default=-1
        The time taken to fit the clusterer.
    predict_time : int, default=-1
        The time taken to predict the cluster labels.
    benchmark_time : int, default=-1
        A benchmark time for the hardware used to scale other timings.
    memory_usage : int, default=-1
        The memory usage of the clusterer.
    n_classes : int, default=-1
        The number of classes in the dataset.
    n_clusters : int, default=-1
        The number of clusters founds by the clusterer.
    """
    if len(cluster_predictions) != cluster_probabilities.shape[0] != len(class_labels):
        raise IndexError(
            "The number of predicted values is not the same as the number of actual "
            "class values."
        )

    if n_clusters > -1 and n_clusters != cluster_probabilities.shape[1]:
        raise IndexError(
            "The number of clusters is not the same as the number of probability "
            "values for each case."
        )

    third_line = (
        f"{clustering_accuracy},"
        f"{fit_time},"
        f"{predict_time},"
        f"{benchmark_time},"
        f"{memory_usage},"
        f"{n_classes},"
        f"{n_clusters}"
    )

    write_results_to_tsml_format(
        cluster_predictions,
        class_labels,
        clusterer_name,
        dataset_name,
        file_path,
        predicted_probabilities=cluster_probabilities,
        full_path=full_path,
        split=split,
        resample_id=resample_id,
        time_unit=time_unit,
        first_line_comment=first_line_comment,
        second_line=parameter_info,
        third_line=third_line,
    )


def write_forecasting_results(
    predictions,
    labels,
    forecaster_name,
    dataset_name,
    file_path,
    full_path=True,
    split=None,
    random_seed=None,
    time_unit="N/A",
    first_line_comment=None,
    parameter_info="No Parameter Info",
    mape=-1,
    fit_time=-1,
    predict_time=-1,
    benchmark_time=-1,
    memory_usage=-1,
):
    """Write the predictions for a forecasting experiment in the format used by tsml.

    Parameters
    ----------
    predictions : np.array
        The predicted values to write to file. Must be the same length as labels.
    labels : np.array
        The actual label values written to file with the predicted values.
    forecaster_name : str
        Name of the forecaster that made the predictions. Written to file and can
        determine file structure if full_path is False.
    dataset_name : str
        Name of the problem the forecaster was built on.
    file_path : str
        Path to write the results file to or the directory to build the default file
        structure if full_path is False.
    full_path : boolean, default=True
        If True, results are written directly to the directory passed in file_path.
        If False, then a standard file structure using the forecaster and dataset names
        is created and used to write the results file.
    split : str or None, default=None
        Either None, 'TRAIN' or 'TEST'. Influences the result file name and first line
        of the file.
    random_seed : int or None, default=None
        Indicates what random seed was used as a random_state for the forecaster.
    time_unit : str, default="N/A"
        The format used for timings in the file, i.e. 'Seconds', 'Milliseconds',
        'Nanoseconds'
    first_line_comment : str or None, default=None
        Optional comment appended to the end of the first line, i.e. the file used to
        generate the results.
    parameter_info : str, default="No Parameter Info"
        Unstructured estimator dependant information, i.e. estimator parameters or
        values from the model build.
    mape: float, default=-1
        The mean absolute percentage error of the predictions.
    fit_time : int, default=-1
        The time taken to fit the forecaster.
    predict_time : int, default=-1
        The time taken to predict the forecasting labels.
    benchmark_time : int, default=-1
        A benchmark time for the hardware used to scale other timings.
    memory_usage : int, default=-1
        The memory usage of the forecaster.
    """
    third_line = (
        f"{mape},"
        f"{fit_time},"
        f"{predict_time},"
        f"{benchmark_time},"
        f"{memory_usage}"
    )

    write_results_to_tsml_format(
        predictions,
        labels,
        forecaster_name,
        dataset_name,
        file_path,
        full_path=full_path,
        split=split,
        resample_id=random_seed,
        time_unit=time_unit,
        first_line_comment=first_line_comment,
        second_line=parameter_info,
        third_line=third_line,
    )


def write_results_to_tsml_format(
    predictions,
    labels,
    estimator_name,
    dataset_name,
    file_path,
    predicted_probabilities=None,
    full_path=True,
    split=None,
    resample_id=None,
    time_unit="N/A",
    first_line_comment=None,
    second_line="No Parameter Info",
    third_line="N/A",
):
    """Write the predictions for an experiment in the standard format used by tsml.

    Parameters
    ----------
    predictions : np.array
        The predicted values to write to file. Must be the same length as labels.
    labels : np.array
        The actual label values written to file with the predicted values. If no label
        is available for a case, a NaN value should be substituted.
    estimator_name : str
        Name of the estimator that made the predictions. Written to file and can
        determine file structure if full_path is False.
    dataset_name : str
        Name of the problem the estimator was built on.
    file_path : str
        Path to write the results file to or the directory to build the default file
        structure if full_path is False.
    predicted_probabilities : np.ndarray, default=None
        Estimated label probabilities. If passed, these are written after the
        predicted values for each case.
    full_path : boolean, default=True
        If True, results are written directly to the directory passed in file_path.
        If False, then a standard file structure using the estimator and dataset names
        is created and used to write the results file.
    split : str or None, default=None
        Either None, 'TRAIN' or 'TEST'. Influences the result file name and first line
        of the file.
    resample_id : int or None, default=None
        Indicates what random seed was used to resample the data or used as a
        random_state for the estimator.
    time_unit : str, default="N/A"
        The format used for timings in the file, i.e. 'Seconds', 'Milliseconds',
        'Nanoseconds'
    first_line_comment : str or None, default=None
        Optional comment appended to the end of the first line, i.e. the file used to
        generate the results or a dictionary linking label indices to actual values.
    second_line : str, default="No Parameter Info"
        Unstructured estimator dependant information, i.e. estimator parameters or
        values from the model build.
    third_line : str, default = "N/A"
        Summary performance information, what values are written depends on the task.
    """
    if len(predictions) != len(labels):
        raise IndexError(
            "The number of predicted values is not the same as the number of actual "
            "labels."
        )

    # If the full directory path is not passed, make the standard structure
    if not full_path:
        file_path = f"{file_path}/{estimator_name}/Predictions/{dataset_name}/"

    os.makedirs(file_path, exist_ok=True)

    if split is None:
        split = ""
    elif split.lower() != "train" and split.lower() != "test":
        raise ValueError("Unknown 'split' value - should be 'TRAIN', 'TEST' or None")

    fname = (
        f"{split.lower()}Results"
        if resample_id is None
        else f"{split.lower()}Resample{resample_id}"
    )
    fname = fname.lower() if split == "" else fname

    with open(f"{file_path}/{fname}.csv", "w") as file:
        # the first line of the output file is in the form of:
        first_line = (
            f"{dataset_name},"
            f"{estimator_name},"
            f"{'No split' if split == '' else split.upper()},"
            f"{'None' if resample_id is None else resample_id},"
            f"{time_unit.upper()},"
            f"{'' if first_line_comment is None else first_line_comment}"
        )
        file.write(first_line + "\n")

        # the second line of the output is free form and estimator-specific; usually
        # this will record info such as paramater options used, any constituent model
        # names for ensembles, etc.
        file.write(str(second_line) + "\n")

        # the third line of the file depends on the task i.e. classification or
        # regression
        file.write(str(third_line) + "\n")

        # from line 4 onwards each line should include the actual and predicted class
        # labels (comma-separated). If present, for each case, the probabilities of
        # predicting every class value for this case should also be appended to the
        # line (a space is also included between the predicted value and the
        # predict_proba). E.g.:
        #
        # if predict_proba data IS provided for case i:
        #   labels[i], preds[i],,prob_class_0[i],
        #   prob_class_1[i],...,prob_class_c[i]
        #
        # if predict_proba data IS NOT provided for case i:
        #   labels[i], preds[i]
        #
        # If labels[i] is NaN (if clustering), labels[i] is replaced with ? to indicate
        # missing
        for i in range(0, len(predictions)):
            label = "?" if np.isnan(labels[i]) else labels[i]
            file.write(f"{label},{predictions[i]}")

            if predicted_probabilities is not None:
                file.write(",")
                for j in predicted_probabilities[i]:
                    file.write(f",{j}")
            file.write("\n")


def _results_present(path, estimator, dataset, resample_id=None, split="TEST"):
    """Check if results are present already."""
    resample_str = "Results" if resample_id is None else f"Resample{resample_id}"
    path = f"{path}/{estimator}/Predictions/{dataset}/"

    if split == "BOTH":
        full_path = f"{path}test{resample_str}.csv"
        full_path2 = f"{path}train{resample_str}.csv"

        if os.path.exists(full_path) and os.path.exists(full_path2):
            return True
    else:
        if split is None or split == "" or split == "NONE":
            full_path = f"{path}{resample_str.lower()}.csv"
        elif split == "TEST":
            full_path = f"{path}test{resample_str}.csv"
        elif split == "TRAIN":
            full_path = f"{path}train{resample_str}.csv"
        else:
            raise ValueError(f"Unknown split value: {split}")

        if os.path.exists(full_path):
            return True

    return False


def fix_broken_second_line(file_path, save_path=None):
    """Fix a results while where the written second line has line breaks.

    This function will remove line breaks from any lines between the first line and the
    first seen valid 'third_line' for any results file format.

    Parameters
    ----------
    file_path : str
        Path to the results file to be fixed, including the file itself.
    save_path : str, default=None
        Path to save the fixed results file to, including the file new files name.
        If None, the new file will replace the original file.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    line_count = 2
    while (
        not _check_classification_third_line(lines[line_count])
        and not _check_regression_third_line(lines[line_count])
        and not _check_clustering_third_line(lines[line_count])
    ):
        if line_count == len(lines) - 1:
            raise ValueError("No valid third line found in input results file.")
        line_count += 1

    if line_count != 2:
        lines[1] = lines[1].replace("\n", " ").replace("\r", " ")
        for i in range(2, line_count - 1):
            lines[1] = lines[1] + lines[i].replace("\n", " ").replace("\r", " ")
        lines[1] = lines[1] + lines[line_count - 1]
        lines = lines[:2] + lines[line_count:]

    if save_path is not None or line_count != 2:
        if save_path is None:
            save_path = file_path

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            f.writelines(lines)


def compare_result_file_resample(file_path1, file_path2):
    """Validate that a two results files use the same data resample.

    Files are deemed as having the same resample if the file length is the same and all
    true label values are the same in both files.

    Parameters
    ----------
    file_path1 : str
        Path to the first results file to be compared, including the file itself.
    file_path1 : str
        Path to the second results file to be compared, including the file itself.

    Returns
    -------
    same_resample : bool
        True if the results file use the same data resample, False otherwise.
    """
    with open(file_path1, "r") as f:
        lines1 = f.readlines()

    with open(file_path2, "r") as f:
        lines2 = f.readlines()

    if len(lines1) != len(lines2):
        raise ValueError("Input results file have different numbers of lines.")

    for i in range(3, len(lines1)):
        if lines1[i].split(",")[0] != lines2[i].split(",")[0]:
            return False

    return True


def assign_gpu(set_environ=False):  # pragma: no cover
    """Assign a GPU to the current process.

    Looks at the available Nvidia GPUs and assigns the GPU with the lowest used memory.

    Parameters
    ----------
    set_environ : bool
        Set the CUDA_DEVICE_ORDER environment variable to "PCI_BUS_ID" anf the
        CUDA_VISIBLE_DEVICES environment variable to the assigned GPU.

    Returns
    -------
    gpu : int
        The GPU assigned to the current process.
    """
    stats = gpustat.GPUStatCollection.new_query()
    pairs = [
        [
            gpu.entry["index"],
            float(gpu.entry["memory.used"]) / float(gpu.entry["memory.total"]),
        ]
        for gpu in stats
    ]

    gpu = min(pairs, key=lambda x: x[1])[0]

    if set_environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    return gpu


def timing_benchmark(num_arrays=1000, array_size=20000, random_state=None):
    """
    Measures the time taken to sort a given number of numpy arrays of a specified size.

    Returns the time taken in milliseconds.

    Parameters
    ----------
    num_arrays: int, default=1000
        Number of arrays to generate and sort.
    array_size: int, default=20000
        Size of each numpy array to be sorted.
    random_state: int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    time_taken: int
        Time taken to sort the arrays in milliseconds.
    """
    if random_state is None:
        random_state = check_random_state(0)
    elif isinstance(random_state, (int, np.random.RandomState)):
        random_state = check_random_state(random_state)
    else:
        raise ValueError("random_state must be an int, RandomState instance or None")

    total_time = 0
    for _ in range(num_arrays):
        array = random_state.rand(array_size)
        start_time = time.time()
        np.sort(array)
        end_time = time.time()
        total_time += end_time - start_time

    return int(round(total_time * 1000))


def estimator_attributes_to_file(
    estimator, dir_path, estimator_name=None, max_depth=np.inf, max_list_shape=np.inf
):
    """Write the attributes of an estimator to file(s).

    Write the attributes of an estimator to file at a given directory. The function
    will recursively write the attributes of any estimators or non-string sequences
    containing estimators found in the attributes of the input estimator to spearate
    files.

    Parameters
    ----------
    estimator : estimator instance
        The estimator to write the attributes of.
    dir_path : str
        The directory to write the attribute files to.
    estimator_name : str or None, default=None
        The name of the estimator. If None, the name of the estimator class will be
        used.
    max_depth : int, default=np.inf
        The maximum depth to go when recursively writing attributes of estimators.
    max_list_shape : int, default=np.inf
        The maximum shape of a list to write when recursively writing attributes of
        contained estimators. i.e. for 0, no estimators contained in lists will be
        written, for 1, only estimators contained in 1-dimensional lists or the top
        level of a list will be written.
    """
    estimator_name = (
        estimator.__class__.__name__ if estimator_name is None else estimator_name
    )
    _write_estimator_attributes_recursive(
        estimator, dir_path, estimator_name, 0, max_depth, max_list_shape
    )


def _write_estimator_attributes_recursive(
    estimator, dir_path, file_name, depth, max_depth, max_list_shape
):
    if depth > max_depth:
        return

    path = f"{dir_path}/{file_name}.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        for attr in estimator.__dict__:
            value = getattr(estimator, attr)
            file.write(f"{attr}: {value}\n")

            if depth + 1 <= max_depth:
                if isinstance(value, BaseEstimator):
                    new_dir_path = f"{dir_path}/{attr}/"
                    file.write(f"    See {new_dir_path}{attr}.txt for more details\n")
                    _write_estimator_attributes_recursive(
                        value, new_dir_path, attr, depth + 1, max_depth, max_list_shape
                    )
                elif _is_non_string_sequence(value):
                    _write_list_attributes_recursive(
                        value,
                        file,
                        dir_path,
                        attr,
                        depth + 1,
                        max_depth,
                        1,
                        max_list_shape,
                    )


def _write_list_attributes_recursive(
    it, file, dir_path, file_name, depth, max_depth, shape, max_list_shape
):
    if shape > max_list_shape:
        return

    for idx, item in enumerate(it):
        if isinstance(item, BaseEstimator):
            new_dir_path = f"{dir_path}/{file_name}_{idx}/"
            file.write(
                f"    See {new_dir_path}{file_name}_{idx}.txt for more details\n"
            )
            _write_estimator_attributes_recursive(
                item,
                new_dir_path,
                f"{file_name}_{idx}",
                depth,
                max_depth,
                max_list_shape,
            )
        elif _is_non_string_sequence(item):
            _write_list_attributes_recursive(
                item,
                file,
                dir_path,
                f"{file_name}_{idx}",
                depth,
                max_depth,
                shape + 1,
                max_list_shape,
            )


def _is_non_string_sequence(obj):
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))
