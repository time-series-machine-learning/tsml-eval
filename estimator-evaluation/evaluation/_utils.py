# -*- coding: utf-8 -*-
import os
import platform
from csv import reader
from operator import itemgetter
from os.path import abspath, join
from typing import Callable, Dict, List, Tuple, TypedDict, Union

import numpy as np
import pandas as pd


# Typing for experiment reading
class ExperimentResamples(TypedDict):
    train_resamples: List[str]
    test_resamples: List[str]


class Dataset(TypedDict):
    dataset_name: str
    resamples: ExperimentResamples


class EstimatorExperiment(TypedDict):
    experiment_name: str
    datasets: List[Dataset]


class Estimator(TypedDict):
    estimator_name: str
    experiment_results: List[EstimatorExperiment]


class Experiment(TypedDict):
    experiment_name: str
    estimators: List[Estimator]


class MetricCallable(TypedDict):
    name: str
    callable: Callable[[List, List], float]


def resolve_experiment_paths(path: str, experiment_name: str) -> Experiment:
    """Resolve the path to an experiment directory.

    Parameters
    ----------
    path: str
        Path to an experiment directory.
    experiment_name: str
        Name of the experiment results reading in.

    Returns
    -------
    List[Estimator]
        List of estimator experiment dicts.
    """
    experiment_path = abspath(path)

    experiment: Experiment = {"experiment_name": experiment_name, "estimators": []}

    # Loop through all subdirectories looking for the 'Predictions' directory
    for subdir, dirs, files in os.walk(experiment_path):
        if "Predictions" in subdir:
            if len(files) > 0:
                for file in files:
                    if "csv" in file:
                        experiment = _add_experiment_result(subdir, file, experiment)

    return experiment


def _find_dict_item(
    dict_list: List[dict], key: str, value: str, create_new_item: dict
) -> dict:
    """Find an item in a list of dictionaries.

    Parameters
    ----------
    dict_list: List[dict]
        List of dictionaries.
    key: str
        Key to search for.
    value: str
        Value to search for.
    create_new_item: dict
        Dictionary to create if the item is not found.

    Returns
    -------
    dict
        The found or added dictionary.
    """

    for curr_dict in dict_list:
        if curr_dict[key] == value:
            return curr_dict

    dict_list.append(create_new_item)
    return create_new_item


def _add_experiment_result(
    subdir: str, file: str, experiment_dict: Experiment
) -> Experiment:
    """Add an experiment result to the experiment dictionary.

    Parameters
    ----------
    subdir: str
        Path to the directory containing the experiment result.
    file: str
        Name of the file containing the experiment result.
    experiment_dict: Experiment
        Dictionary containing the experiment results.

    Returns
    -------
    Experiment
        Updated dictionary containing the experiment results.
    """
    # If on windows use different split
    if "Windows" in platform.platform():
        split_subdir = subdir.split("\\")
    else:
        split_subdir = subdir.split("/")

    with open(join(subdir, file), "r") as f:
        first_line = (f.readline()).split(",")

    curr_estimator_name = split_subdir[-4]
    curr_experiment_name = first_line[1]
    curr_dataset_name = first_line[0]

    estimator: Estimator = _find_dict_item(
        experiment_dict["estimators"],
        "estimator_name",
        curr_estimator_name,
        {"estimator_name": curr_estimator_name, "experiment_results": []},
    )

    estimator_experiment: EstimatorExperiment = _find_dict_item(
        estimator["experiment_results"],
        "experiment_name",
        curr_experiment_name,
        {"experiment_name": curr_experiment_name, "datasets": []},
    )

    dataset: Dataset = _find_dict_item(
        estimator_experiment["datasets"],
        "dataset_name",
        curr_dataset_name,
        {
            "dataset_name": curr_dataset_name,
            "resamples": {"train_resamples": [], "test_resamples": []},
        },
    )

    if file.startswith("train"):
        dataset["resamples"]["train_resamples"].append(join(subdir, file))
    elif file.startswith("test"):
        dataset["resamples"]["test_resamples"].append(join(subdir, file))
    else:
        raise ValueError("File name must start with train or test")

    return experiment_dict


def _extract_resamples_for_dataset(
    dataset: Dataset, key: str, metric_callables: List[MetricCallable]
) -> Dict:
    """Creates a row for each of the metrics for each resample.

    Parameters
    ----------
    dataset: Dataset
        Dataset to extract resamples for.
    key: str
        Key to use for the resample name. Must be either 'train_resamples' or
        'test_resamples'.
    metric_callables: List[MetricCallable]
        List of metric callables to use.

    Returns
    -------
    Dict
        Dictionary containing the resample names and the metric values. The dict
        takes the form of:
        {
            'metric_name': [resample1, resample2, ...]
        }
    """
    if key != "train_resamples" and key != "test_resamples":
        raise ValueError("key must be train_resamples or test_resamples")

    resamples = []
    for resample in dataset["resamples"][key]:
        resamples.append(_csv_results_to_metric(resample, metric_callables))

    resample_row = {}
    for resample in resamples:
        for metric_name, metric_value in resample.items():
            if metric_name not in resample_row:
                resample_row[metric_name] = []
            resample_row[metric_name].append(metric_value)

    return resample_row


def _check_equal_resamples(metric_rows: List) -> List:
    """Check if results all have same number of resamples.

    If they don't have same number of resamples take the most common amount of equal
    resamples.

    Parameters
    ----------
    metric_rows: List
        List of metric rows.

    Returns
    -------
    List
        Metric row where all have same number of resamples
    """
    extra_row_temp = {}
    for i in range(len(metric_rows)):
        row = metric_rows[i]
        if str(len(row)) not in extra_row_temp:
            extra_row_temp[str(len(row))] = []
        extra_row_temp[str(len(row))].append(i)

    max = 0
    found = ""
    for key in extra_row_temp:
        if len(extra_row_temp[key]) > max:
            found = key
            max = len(extra_row_temp[key])

    metric_rows = list(itemgetter(*extra_row_temp[found])(metric_rows))
    return metric_rows


def extract_estimator_experiment(
    estimator_experiment: EstimatorExperiment, metric_callables: List[MetricCallable]
) -> Tuple[List[Tuple[str, pd.DataFrame]], List[Tuple[str, pd.DataFrame]]]:
    """Extract the results of an estimator experiment.

    Each data in the list of dataframes return will take the form:
    | folds    | 0   | 1   | 2   | 3   | ... |
    ------------------------------------------
    | dataset1 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
    | dataset2 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
    | dataset3 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
    | dataset4 | 0.0 | 0.0 | 0.0 | 0.0 | ... |

    Parameters
    ----------
    estimator_experiment: EstimatorExperiment
        The estimator experiment to extract the results from.
    metric_callables: List[MetricCallable]
        List of metric callables to use to evaluate the results.

    Returns
    -------
    List[Tuple[str, pd.DataFrame]]
        List of tuples containing the metric name as first element and dataframe
        containing the metric train results.
    List[Tuple[str, pd.DataFrame]]
        List of tuples containing the metric name as first element and dataframe
        containing the metric test results.
    """

    train_results = []
    test_results = []
    result_dict = {"train": {}, "test": {}}

    # Function that takes metrics and organises it into rows
    def create_metric_rows(resamples: Dict, dataset_name: str, split: str):
        curr = result_dict[split]
        for metric_name, metric_values in resamples.items():
            if metric_name not in curr:
                curr[metric_name] = []
            curr[metric_name].append([dataset_name] + metric_values)

    # Loops through each dataset and each resample metric is organised into a row
    for dataset in estimator_experiment["datasets"]:
        train_resamples = _extract_resamples_for_dataset(
            dataset, "train_resamples", metric_callables
        )
        test_resamples = _extract_resamples_for_dataset(
            dataset, "test_resamples", metric_callables
        )

        create_metric_rows(train_resamples, dataset["dataset_name"], "train")
        create_metric_rows(test_resamples, dataset["dataset_name"], "test")

    # Function creates dataframe from a row and adds column headings
    def create_df(split: str):
        curr = result_dict[split]
        temp = []
        for metric_name, metric_rows in curr.items():

            metric_rows = _check_equal_resamples(metric_rows)
            columns = ["folds"] + list(range(0, len(metric_rows[0]) - 1))

            temp.append((metric_name, pd.DataFrame(metric_rows, columns=[columns])))
        return temp

    train_dfs = create_df("train")
    test_dfs = create_df("test")
    return train_dfs, test_dfs


def read_results_from_uea_format(
    path: str,
    meta_col_headers: List[str] = None,
    prediction_col_headers: List[str] = None,
) -> Dict:
    """Read results from uea format.

    Parameters
    ----------
    path: str
        Path to results file csv.
    meta_col_headers: List[str], defaults = None
        Column header for meta data about estimator (third line)
    prediction_col_headers: List[str], defaults = None
        Column header for predictions data (fourth line and onwards)

    Returns
    -------
    dict
        Dict in the following format:
        {
            'first_line_comment': [first line data]
            'estimator_parameters': [second line data]
            'estimator_meta': [third line data]
            'predictions': [forth line and onwards]
        }

    """
    read_dict = {}
    with open(path, "r") as read_obj:
        csv_reader = reader(read_obj)

        read_dict["first_line_comment"] = next(csv_reader)
        read_dict["estimator_parameters"] = next(csv_reader)

        read_dict["estimator_meta"] = []
        if meta_col_headers is not None:
            read_dict["estimator_meta"].append(meta_col_headers)
        read_dict["estimator_meta"].append(next(csv_reader))

        read_dict["predictions"] = []

        if prediction_col_headers is not None:
            read_dict["predictions"].append(prediction_col_headers)

        for row in csv_reader:
            read_dict["predictions"].append(row)

    return read_dict


def read_clusterer_result_from_uea_format(csv_path):
    meta_col_headers = [
        "N/A",
        "build time",
        "test time",
        "N/A",
        "N/A",
        "num classes",
        "num classes",
    ]
    with open(csv_path, "r") as read_obj:
        csv_reader = reader(read_obj)
        curr_line = next(csv_reader)
        while "}" not in curr_line[-1]:
            curr_line = next(csv_reader)
        meta = next(csv_reader)  # Skip second line
        num_classes = meta[-1]

    prediction_col_headers = ["True y class", "Predicted y class", "N/A"]

    for i in range(1, int(num_classes) + 1):
        prediction_col_headers.append(f"proba of class {str(i)}")

    return read_results_from_uea_format(
        csv_path, meta_col_headers, prediction_col_headers
    )


def _csv_results_to_metric(
    csv_path: str, metric_callables: List[MetricCallable]
) -> Dict:
    """Read results from csv and return a dict of metric results.

    Parameters
    ----------
    csv_path: str
        Path to csv file containing the results.
    metric_callables: List[MetricCallable]
        List of metric callables to use to evaluate the results.

    Returns
    -------
    dict
        Dict of metric results. The dict will have the following format:
        {
            'metric_name': metric_value,
        }
    """
    data = read_clusterer_result_from_uea_format(csv_path)["predictions"]

    # TODO: update the experiment method so it can't ever read in arrays but for now
    # we filter out the dict
    remove_start = 0
    for i in range(len(data)):
        val = data[i]
        for string in val:
            if "}" in string:
                remove_start = i

    data = data[remove_start:-1]

    columns = data[0][0:2]

    # Read in only first two columns which is true class followed by predicted.
    true_label = []
    predicted_label = []
    for val in data[1:]:
        true_label.append(int(val[0]))
        predicted_label.append(int(val[1]))

    metric_results = {}
    for metric in metric_callables:
        metric_name = metric["name"]
        metric_callable = metric["callable"]

        if metric_name not in metric_results:
            metric_results[metric_name] = 0.0

        metric_results[metric_name] = metric_callable(true_label, predicted_label)

    return metric_results


class EstimatorMetricResults(TypedDict):
    estimator_name: str
    result: pd.DataFrame


class MetricResults(TypedDict):
    metric_name: str
    test_estimator_results: Union[List[EstimatorMetricResults], None]
    train_estimator_results: Union[List[EstimatorMetricResults], None]


def read_metric_results(path: str) -> List[str]:
    """Read results from csv and return a dict of metric results.

    Parameters
    ----------
    path: str
        Path to directory containing csv results.

    Returns
    -------
    List[str]
        List of path to metric results csv files.
    """
    result_path = abspath(path)
    test = []
    for subdir, dirs, files in os.walk(result_path):
        if len(files) > 0:
            for file in files:
                if "csv" in file:
                    test.append(join(subdir, file))

    return test


def _split_metric_result_to_summary(
    result: List[MetricResults], split: str = "test_estimator_results"
):
    column_headers = ["estimator", "dataset"]
    temp_dict = {}
    for metric_result in result:
        curr_metric = metric_result["metric_name"]
        column_headers.append(curr_metric)

        for test_result in metric_result[split]:
            curr_estimator = test_result["estimator_name"]
            data = test_result["result"].copy()

            dataset_col = list(data[data.columns[0]][0:])

            del data[data.columns[0]]
            data = data.mean(axis=1)

            estimator_col = [curr_estimator] * (len(dataset_col))

            curr_df = pd.DataFrame(
                [estimator_col, dataset_col, data],
            ).T

            for index, row in curr_df.iterrows():
                name = f"{row[0]}:::{row[1]}"
                if name not in temp_dict:
                    temp_dict[name] = []
                temp_dict[name].append((curr_metric, row[2]))

    df_list = []

    for key, value in temp_dict.items():
        estimator_name, dataset_name = key.split(":::")
        row = [estimator_name, dataset_name]
        for metric_name in column_headers[2:]:
            for metric in value:
                curr_metric = metric[0]
                if metric_name == curr_metric:
                    metric_value = metric[1]
                    row.append(metric_value)
                    break
        df_list.append(row)

    df = pd.DataFrame(df_list, columns=column_headers)
    return df


def metric_result_to_summary(
    result: List[MetricResults], split: str = "both"
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Convert metric result to data frame of the format:
        ----------------------------------
        | estimator | dataset | metric1  | metric2 |
        | cls1      | data1   | 1.2      | 1.2     |
        | cls2      | data2   | 3.4      | 1.4     |
        | cls1      | data2   | 1.4      | 1.3     |
        | cls2      | data1   | 1.3      | 1.2     |
        ----------------------------------

    Parameters
    ----------
    result: List[MetricResults]
        Metric results to convert.
    split: str, default='both'
        Whether to split the results into train and test results. If 'both' then
        both train and test results will be returned. If 'test' then only test
        results will be returned. If 'train' then only train results will be
        returned.

    Returns
    -------
    pd.DataFrame
        Data frame with metric results converted for test data. This will be of the format:
        ----------------------------------
        | estimator | dataset | metric1  | metric2 |
        | cls1      | data1   | 1.2      | 1.2     |
        | cls2      | data2   | 3.4      | 1.4     |
        | cls1      | data2   | 1.4      | 1.3     |
        | cls2      | data1   | 1.3      | 1.2     |
        ----------------------------------
    pd.DataFrame
        Data frame with metric results converted for training data. This will be of the format:
        ----------------------------------
        | estimator | dataset | metric1  | metric2 |
        | cls1      | data1   | 1.2      | 1.2     |
        | cls2      | data2   | 3.4      | 1.4     |
        | cls1      | data2   | 1.4      | 1.3     |
        | cls2      | data1   | 1.3      | 1.2     |
        ----------------------------------
    """
    test_data = _split_metric_result_to_summary(result, split="test_estimator_results")
    train_data = _split_metric_result_to_summary(
        result, split="train_estimator_results"
    )
    if split == "test":
        return test_data
    elif split == "train":
        return train_data
    else:
        return test_data, train_data


def from_metric_summary_to_dataset_format(
    summary_format: pd.DataFrame, return_numpy: np.ndarray = False
) -> Union[pd.DataFrame, np.ndarray, List[pd.DataFrame], List[np.ndarray]]:
    """Converts summary format to dataset format.

    Parameters
    ----------
    summary_format: pd.DataFrame
        Summary format dataframe.

    Returns
    -------
    Union[pd.DataFrame, np.ndarray, List[pd.DataFrame], List[np.ndarray]]:
        Dataset formatted data. If multiple metrics will return list if only one then
        only that is returned.
    """
    metrics = list(set(summary_format.columns[2:]))
    datasets = list(summary_format["dataset"].unique())
    classifiers = list(summary_format["estimator"].unique())

    return_result = []
    for curr_metric in metrics:
        metric_index = summary_format.columns.get_loc(curr_metric)
        columns = ["Problem"] + classifiers
        rows = []
        for dataset in datasets:
            rows.append([dataset] + ([None] * len(classifiers)))

        for i in range(len(classifiers)):
            estimator = classifiers[i]
            curr_df = summary_format[summary_format["estimator"] == estimator]
            for j in range(len(datasets)):
                curr_dataset = datasets[j]
                curr_metric = curr_df[curr_df["dataset"] == datasets[j]]
                try:
                    curr_res = curr_metric.iloc[0, metric_index]
                except:
                    continue

                rows[j][i + 1] = curr_res
        curr = pd.DataFrame(rows, columns=columns)
        if return_numpy is True:
            temp = curr.to_numpy()
            col_header = np.array([curr.columns.to_numpy()])
            return_result.append(np.concatenate((col_header, temp), axis=0))
        else:
            return_result.append(curr)

    if len(return_result) == 1:
        return return_result[0]
    return return_result


def from_metric_dataset_format_to_metric_summary(
    summary_format: pd.DataFrame, return_numpy: np.ndarray = False
) -> Union[pd.DataFrame, np.ndarray, List[pd.DataFrame], List[np.ndarray]]:
    """Converts summary format to dataset format.

    Parameters
    ----------
    summary_format: pd.DataFrame
        Summary format dataframe.

    Returns
    -------
    Union[pd.DataFrame, np.ndarray, List[pd.DataFrame], List[np.ndarray]]:
        Dataset formatted data. If multiple metrics will return list if only one then
        only that is returned.
    """
    metrics = list(set(summary_format.columns[2:]))
    datasets = list(summary_format["dataset"].unique())
    classifiers = list(summary_format["estimator"].unique())

    return_result = []
    for curr_metric in metrics:
        metric_index = summary_format.columns.get_loc(curr_metric)
        columns = ["Problem"] + classifiers
        rows = []
        for dataset in datasets:
            rows.append([dataset] + ([None] * len(classifiers)))

        for i in range(len(classifiers)):
            estimator = classifiers[i]
            curr_df = summary_format[summary_format["estimator"] == estimator]
            for j in range(len(datasets)):
                curr_dataset = datasets[j]
                curr_metric = curr_df[curr_df["dataset"] == datasets[j]]
                try:
                    curr_res = curr_metric.iloc[0, metric_index]
                except:
                    continue

                rows[j][i + 1] = curr_res
        curr = pd.DataFrame(rows, columns=columns)
        if return_numpy is True:
            temp = curr.to_numpy()
            col_header = np.array([curr.columns.to_numpy()])
            return_result.append(np.concatenate((col_header, temp), axis=0))
        else:
            return_result.append(curr)

    if len(return_result) == 1:
        return return_result[0]
    return return_result


def combine_two_summary_df(first_df: pd.DataFrame, second_df: pd.DataFrame):
    first_cols = set(first_df.columns)
    second_cols = set(second_df.columns)
    drop_cols = first_cols.difference(second_cols)

    if drop_cols in first_cols:
        first_df = first_df.drop(columns=drop_cols)
    if drop_cols in second_cols:
        second_df = second_df.drop(columns=drop_cols)

    return pd.concat([first_df, second_df])
