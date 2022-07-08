from typing import List, TypedDict, Callable, Tuple, Dict, Union
import os
from os.path import abspath, join
from csv import reader
from operator import itemgetter
import platform

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

    experiment: Experiment = {
        'experiment_name': experiment_name,
        'estimators': []
    }

    # Loop through all subdirectories looking for the 'Predictions' directory
    for subdir, dirs, files in os.walk(experiment_path):
        if 'Predictions' in subdir:
                if len(files) > 0:
                    for file in files:
                        if 'csv' in file:
                            experiment = _add_experiment_result(subdir, file, experiment)

    return experiment


def _find_dict_item(
        dict_list: List[dict],
        key: str,
        value: str,
        create_new_item: dict
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
        subdir: str,
        file: str,
        experiment_dict: Experiment
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
    if 'Windows' in platform.platform():
        split_subdir = subdir.split('\\')
    else:
        split_subdir = subdir.split('/')

    with open(join(subdir, file), 'r') as f:
        first_line = (f.readline()).split(',')

    curr_estimator_name = split_subdir[-4]
    curr_experiment_name = first_line[1]
    curr_dataset_name = first_line[0]

    estimator: Estimator = _find_dict_item(
        experiment_dict['estimators'],
        'estimator_name',
        curr_estimator_name,
        {
            'estimator_name': curr_estimator_name,
            'experiment_results': []
        }
    )

    estimator_experiment: EstimatorExperiment = _find_dict_item(
        estimator['experiment_results'],
        'experiment_name',
        curr_experiment_name,
        {
            'experiment_name': curr_experiment_name,
            'datasets': []
        }
    )

    dataset: Dataset = _find_dict_item(
        estimator_experiment['datasets'],
        'dataset_name',
        curr_dataset_name,
        {
            'dataset_name': curr_dataset_name,
            'resamples':
                {'train_resamples': [], 'test_resamples': []}
        }
    )

    if file.startswith('train'):
        dataset['resamples']['train_resamples'].append(join(subdir, file))
    elif file.startswith('test'):
        dataset['resamples']['test_resamples'].append(join(subdir, file))
    else:
        raise ValueError('File name must start with train or test')

    return experiment_dict

def _extract_resamples_for_dataset(
        dataset: Dataset,
        key: str,
        metric_callables: List[MetricCallable]
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
    if key != 'train_resamples' and key != 'test_resamples':
        raise ValueError('key must be train_resamples or test_resamples')

    resamples = []
    for resample in dataset['resamples'][key]:
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
    found = ''
    for key in extra_row_temp:
        if len(extra_row_temp[key]) > max:
            found = key
            max = len(extra_row_temp[key])

    metric_rows = list(itemgetter(*extra_row_temp[found])(metric_rows))
    return metric_rows


def extract_estimator_experiment(
        estimator_experiment: EstimatorExperiment,
        metric_callables: List[MetricCallable]
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
    result_dict = {
        'train': {},
        'test': {}
    }

    # Function that takes metrics and organises it into rows
    def create_metric_rows(resamples: Dict, dataset_name: str, split: str):
        curr = result_dict[split]
        for metric_name, metric_values in resamples.items():
            if metric_name not in curr:
                curr[metric_name] = []
            curr[metric_name].append([dataset_name] + metric_values)

    # Loops through each dataset and each resample metric is organised into a row
    for dataset in estimator_experiment['datasets']:
        train_resamples = _extract_resamples_for_dataset(
            dataset, 'train_resamples', metric_callables
        )
        test_resamples = _extract_resamples_for_dataset(
            dataset, 'test_resamples', metric_callables
        )

        create_metric_rows(train_resamples, dataset['dataset_name'], 'train')
        create_metric_rows(test_resamples, dataset['dataset_name'], 'test')

    # Function creates dataframe from a row and adds column headings
    def create_df(split: str):
        curr = result_dict[split]
        temp = []
        for metric_name, metric_rows in curr.items():

            metric_rows = _check_equal_resamples(metric_rows)
            columns = ['folds'] + list(range(0, len(metric_rows[0]) - 1))

            temp.append(
                (
                    metric_name,
                    pd.DataFrame(metric_rows, columns=[columns])
                )
            )
        return temp

    train_dfs = create_df('train')
    test_dfs = create_df('test')
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
        next(csv_reader)  # Skip first line
        next(csv_reader)  # Skip second line
        meta = next(csv_reader)  # Skip second line
        num_classes = meta[-1]

    prediction_col_headers = ["True y class", "Predicted y class", "N/A"]

    for i in range(1, int(num_classes) + 1):
        prediction_col_headers.append(f"proba of class {str(i)}")

    return read_results_from_uea_format(
        csv_path, meta_col_headers, prediction_col_headers
    )


def _csv_results_to_metric(
    csv_path: str,
    metric_callables: List[MetricCallable]
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

    columns = data[0][0:2]

    # Read in only first two columns which is true class followed by predicted.
    true_label = []
    predicted_label = []
    for val in data[1:]:
        true_label.append(int(val[0]))
        predicted_label.append(int(val[1]))

    metric_results = {}
    for metric in metric_callables:
        metric_name = metric['name']
        metric_callable = metric['callable']

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
                if 'csv' in file:
                    test.append(join(subdir, file))

    return test
