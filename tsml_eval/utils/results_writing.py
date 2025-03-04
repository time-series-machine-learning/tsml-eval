"""Utility functions for results writing."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]

__all__ = [
    "write_classification_results",
    "write_regression_results",
    "write_clustering_results",
    "write_forecasting_results",
    "write_results_to_tsml_format",
]

import os

import numpy as np


def write_classification_results(
    predictions,
    probabilities,
    class_labels,
    classifier_name,
    dataset_name,
    file_path,
    full_path=True,
    first_line_classifier_name=None,
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
    first_line_classifier_name : str or None, default=None
        Alternative name for the classifier to be written to the file. If None, the
        classifier_name is used. Useful if full_path is False and extra information is
        wanted in the classifier name (i.e. and alias and class name)
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
        first_line_estimator_name=first_line_classifier_name,
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
    first_line_regressor_name=None,
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
    first_line_regressor_name : str or None, default=None
        Alternative name for the regressor to be written to the file. If None, the
        regressor_name is used. Useful if full_path is False and extra information is
        wanted in the regressor name (i.e. and alias and class name)
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
        first_line_estimator_name=first_line_regressor_name,
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
    first_line_clusterer_name=None,
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
    first_line_clusterer_name : str or None, default=None
        Alternative name for the clusterer to be written to the file. If None, the
        clusterer_name is used. Useful if full_path is False and extra information is
        wanted in the clusterer name (i.e. and alias and class name)
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
        first_line_estimator_name=first_line_clusterer_name,
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
    first_line_forecaster_name=None,
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
    first_line_forecaster_name : str or None, default=None
        Alternative name for the forecaster to be written to the file. If None, the
        forecaster_name is used. Useful if full_path is False and extra information is
        wanted in the forecaster name (i.e. and alias and class name)
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
        first_line_estimator_name=first_line_forecaster_name,
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
    first_line_estimator_name=None,
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
    first_line_estimator_name : str or None, default=None
        Alternative name for the estimator to be written to the file. If None, the
        estimator_name is used. Useful if full_path is False and extra information is
        wanted in the estimator name (i.e. and alias and class name)
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
        d = (
            ""
            if dataset_name is None or dataset_name == "" or dataset_name == "N/A"
            else f"{dataset_name}/"
        )
        file_path = f"{file_path}/{estimator_name}/Predictions/{d}"

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

    if first_line_estimator_name is None:
        first_line_estimator_name = estimator_name

    with open(f"{file_path}/{fname}.csv", "w") as file:
        # the first line of the output file is in the form of:
        first_line = (
            f"{dataset_name},"
            f"{first_line_estimator_name},"
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
