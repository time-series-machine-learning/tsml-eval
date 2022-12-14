# -*- coding: utf-8 -*-
import os
import platform
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    jaccard_score,
    log_loss,
    mutual_info_score,
    normalized_mutual_info_score,
    precision_score,
    rand_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from tsml_eval.evaluation._utils import (
    EstimatorMetricResults,
    Experiment,
    MetricCallable,
    MetricResults,
    extract_estimator_experiment,
    read_metric_results,
    resolve_experiment_paths,
)

CLUSTER_METRIC_CALLABLES = [
    MetricCallable(name="RI", callable=rand_score),
    MetricCallable(name="AMI", callable=adjusted_mutual_info_score),
    MetricCallable(name="NMI", callable=normalized_mutual_info_score),
    MetricCallable(name="ARI", callable=adjusted_rand_score),
    MetricCallable(name="MI", callable=mutual_info_score),
    MetricCallable(name="ACC", callable=accuracy_score),
]

CLASSIFICATION_METRIC_CALLABLES = [
    MetricCallable(name="ACC", callable=accuracy_score),
    MetricCallable(name="F1", callable=f1_score),
    MetricCallable(name="Precision", callable=precision_score),
    MetricCallable(name="Recall", callable=recall_score),
    MetricCallable(name="Jaccard", callable=jaccard_score),
    MetricCallable(name="ROC_AUC", callable=roc_auc_score),
    MetricCallable(name="Brier", callable=brier_score_loss),
    MetricCallable(name="Log_Loss", callable=log_loss),
    MetricCallable(name="Balanced_Accuracy", callable=balanced_accuracy_score),
    MetricCallable(name="Top_k_Accuracy", callable=top_k_accuracy_score),
    MetricCallable(name="Average_Precision", callable=average_precision_score),
]


def _get_metric_callable(
    metric: Union[str, MetricCallable],
) -> MetricCallable:
    """Get the metric callable for the given metric.

    Parameters
    ----------
    metric: Union[str, Callable]
        Metric to get the callable for. If a string is provided, it must be a valid
        metric found in known_metrics.

    Returns
    -------
    MetricCallable
        Metric callable for the given metric.
    """

    if isinstance(metric, str):
        for curr_metric in CLUSTER_METRIC_CALLABLES:
            if curr_metric["name"] == metric:
                return curr_metric
        for curr_metric in CLASSIFICATION_METRIC_CALLABLES:
            if curr_metric.name == metric:
                return curr_metric
        raise ValueError(f"Metric {metric} not found.")
    elif "name" in metric and "callable" in metric:
        return MetricCallable(**metric)
    else:
        raise ValueError(
            f"Metric {metric} not found. It must either be a metric"
            f"callable (for example: "
            f"MetricCallable(name='ARI', callable=adjusted_rand_score)"
            f"or a string that is valid metric name. You can import "
            f"MetricCallable like so: "
            f"from "
            f"sktime_estimator_evaluation.new_evaluation.evaluation.utils "
            f"import MetricCallable"
        )


def _resolve_output_dir(output_dir: str) -> Union[str, None]:
    """Resolve the output directory.

    If it doesn't exist then it is created.

    Parameters
    ----------
    output_dir: str
        Path to the output directory.

    Returns
    -------
    str
        Path to the output directory.
    """
    if output_dir is None:
        return output_dir
    path = os.path.abspath(output_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def _format_metric_results(
    metric_results: List[Tuple[str, pd.DataFrame]],
    estimator_name: str,
    formatted_metric_results: List[MetricResults],
    train_split: bool = True,
):
    """Format the metric results.

    Parameters
    ----------
    metric_results: List[Tuple[str, pd.DataFrame]]
        List of metric results.
    estimator_name: str
        Name of the estimator.
    formatted_metric_results: List[MetricResults]
        List of formatted metric results to store output.
    train_split: bool, defaults = True
        Whether the metric results are for the training split or the test split.
        Should be true if it is train split and false if test split.
    """
    for metric in metric_results:
        metric_name, metric_result = metric
        found_metric_result = None
        for curr_metric_result in formatted_metric_results:
            if curr_metric_result["metric_name"] == metric_name:
                found_metric_result = curr_metric_result
                break

        if found_metric_result is None:
            found_metric_result = {
                "metric_name": metric_name,
                "test_estimator_results": [],
                "train_estimator_results": [],
            }
            formatted_metric_results.append(found_metric_result)

        if train_split is True:
            found_metric_result["train_estimator_results"].append(
                EstimatorMetricResults(
                    estimator_name=estimator_name, result=metric_result
                )
            )
        else:
            found_metric_result["test_estimator_results"].append(
                EstimatorMetricResults(
                    estimator_name=estimator_name, result=metric_result
                )
            )


def evaluate_raw_results(
    experiment_name: str,
    path: str,
    metrics: List[Union[MetricCallable, str]] = None,
    output_dir: str = None,
) -> List[MetricResults]:
    """Evaluate the results of a experiment.

    If output_dir is specified then a dir of the following format will be written at
    the specified location:
    -> estimator1
        -> experiment_name1
            -> test
                -> METRIC_NAME1.csv
                -> METRIC_NAME2.csv
                -> ...
            -> train
                -> METRIC_NAME1.csv
                -> METRIC_NAME2.csv
                -> ...
        -> experiment_name2
        -> experiment_name3
        -> ...
    -> estimator2
    -> ...

    Each csv will contain a table of the following format:
    | folds    | 0   | 1   | 2   | 3   | ... |
    ------------------------------------------
    | dataset1 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
    | dataset2 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
    | dataset3 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
    | dataset4 | 0.0 | 0.0 | 0.0 | 0.0 | ... |

    Parameters
    ----------
    experiment_name: str
        Name of the experiment.
    path: str
        Path to a directory containing the results of an experiment. This
        should be structured as follows:
        -> experiment_directory
            -> experiment_name
                -> estimator_name
                    -> experiment_name
                        -> Predictions (note this must be called 'Predictions')
                            -> dataset_name
                                -> testResample0.csv (must be called test or train)
                                -> testResample1.csv
                                -> testResample2.csv
                                -> testResample3.csv
                                -> trainResample0.csv
                                -> trainResample1.csv
                                -> trainResample2.csv
                                -> trainResample3.csv
    metrics: List[Union[MetricCallable, str]], defaults = None
        List of metrics to evaluate. If None, then only accuracy used. If string then
        list of valid classification metrics is: ['ACC', 'F1', 'Precision', 'Recall',
        'Jacard', 'ROC_AUC', 'Brier', 'Log_Loss', 'Balanced_Accuracy',
        'Top_k_Accuracy', 'Average_Precision']
        A list of valid clustering metrics is:
        ['RI', 'AMI', 'ACC', 'NMI', 'ARI', 'MI']
        If MetricCallable is used then it must be a MetricCallable object. The callable
        must accept two lists the first being the 'true labels' and the second being the
        'predicted labels'. For example:
        def example_metric(true_labels, predicted_labels):
            ...do some computation
            return metric_value
        To create a MetricCallable we pass either:
        MetricCallable(name='example_metric', callable=example_metric)
        or
        {'name': 'example_metric', 'callable': example_metric}
    output_dir: str
        Path to directory to output result. If it doesn't exist then it is created
    split: str, defaults = 'both'
        Whether to evaluate the results for the test split, train split or both.
        Must be 'train', 'test' or 'both'.

    Returns
    -------
    List[Dict]
        A list of metric results. Each metric will take the form:
        {
            'metric_name': str,
            'test_estimator_results': [
                {
                    'estimator_name': str,
                    'result': pd.DataFrame
                },
                {
                    'estimator_name': str,
                    'result': pd.DataFrame
                }
            ],
            'train_estimator_results': [
                {
                    'estimator_name': str,
                    'result': pd.DataFrame
                },
                {
                    'estimator_name': str,
                    'result': pd.DataFrame
                }
            ],
        }
        Each result dataframe will be in the following format:
        | folds    | 0   | 1   | 2   | 3   | ... |
        ------------------------------------------
        | dataset1 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
        | dataset2 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
        | dataset3 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
        | dataset4 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
    """
    if metrics is None:
        metrics = [MetricCallable(name="ACC", callable=accuracy_score)]

    output_dir = _resolve_output_dir(output_dir)

    metrics = [_get_metric_callable(metric) for metric in metrics]
    result: Experiment = resolve_experiment_paths(path, experiment_name)

    def resolve_join_path(curr_path: str, join_to: str) -> Union[str]:
        """Resolve the path to join to."""
        temp_path = None
        if output_dir is not None:
            temp_path = _resolve_output_dir(os.path.join(curr_path, join_to))
        return temp_path

    def write_csv_metric(
        curr_path: str, split: str, metric_results: List[Tuple[str, pd.DataFrame]]
    ):
        """Write the metric results to a csv."""
        inner_path = resolve_join_path(curr_path, split)
        for i in range(0, len(metric_results)):
            metric_name, metric_result = metric_results[i]
            metric_result.to_csv(
                os.path.join(inner_path, f"{metric_name}.csv"), index=False
            )

    formatted_metric_results: List[MetricResults] = []
    estimator_result = {}
    # For each estimator
    for estimator in result["estimators"]:
        print("evaluating estimator: ", estimator["estimator_name"])
        estimator_output_path = resolve_join_path(
            output_dir, estimator["estimator_name"]
        )
        estimator_result[estimator["estimator_name"]] = {}

        # For each experiment for estimator (i.e. hyperparams)
        for experiment in estimator["experiment_results"]:
            print("----> evaluating experiment: ", experiment["experiment_name"])
            estimator_result[experiment["experiment_name"]] = {}
            experiment_output_path = resolve_join_path(
                estimator_output_path, experiment["experiment_name"]
            )

            train_metrics, test_metrics = extract_estimator_experiment(
                experiment, metrics
            )

            if estimator_output_path is not None:
                write_csv_metric(experiment_output_path, "train", train_metrics)
                write_csv_metric(experiment_output_path, "test", test_metrics)

            _format_metric_results(
                train_metrics,
                experiment["experiment_name"],
                formatted_metric_results,
                train_split=True,
            )
            _format_metric_results(
                test_metrics,
                experiment["experiment_name"],
                formatted_metric_results,
                train_split=False,
            )

    return formatted_metric_results


def _default_format_reader(path: str) -> Tuple[str, str, str]:
    """Default format reader.

    Parameters
    ----------
    path: str
        Path to result file.

    Returns
    -------
    str
        Estimator name.
    str
        Metric name.
    str
        Experiment name.
    """
    if "Windows" in platform.platform():
        split_subdir = path.split("\\")
    else:
        split_subdir = path.split("/")
    metric_name = split_subdir[-1].split(".")[0]
    split = split_subdir[-2]
    estimator_name = split_subdir[-3]
    return estimator_name, metric_name, split


def evaluate_metric_results(
    path: str, name_metric_callable: Callable[[str], Tuple[str, str, str]] = None
) -> List[MetricResults]:
    """Read the evaluation metric results from a directory.

    Each csv withing the path directory should be of them format:
        | folds    | 0   | 1   | 2   | 3   | ... |
        ------------------------------------------
        | dataset1 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
        | dataset2 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
        | dataset3 | 0.0 | 0.0 | 0.0 | 0.0 | ... |
        | dataset4 | 0.0 | 0.0 | 0.0 | 0.0 | ... |

    Parameters
    ----------
    path: str
        Path to a directory containing the results of an experiment. This directory
        should contain csvs that are metrics analysis of a given estimator.
    name_metric_callable: Callable[[str], str], defaults = None
        Function that takes the path to a csv and should return the estimator name as
        the first return value, the metric name as the second return value, and
        the split (i.e.'test' or 'train') as the third. It is
        up to you to define how the name and metric used is derived from the path to
        the csv. If none is specified then the default reader that assumes the
        standard format (i.e. created using evaluate_results) is used to derive the
        name and metric.

    Returns
    -------
    Returns
    -------
    List[Dict]
        A list of metric results. Each metric will take the form:
        {
            'metric_name': str,
            'test_estimator_results': [
                {
                    'estimator_name': str,
                    'result': pd.DataFrame
                },
                {
                    'estimator_name': str,
                    'result': pd.DataFrame
                }
            ],
            'train_estimator_results': [
                {
                    'estimator_name': str,
                    'result': pd.DataFrame
                },
                {
                    'estimator_name': str,
                    'result': pd.DataFrame
                }
            ],
        }
    """
    if name_metric_callable is None:
        name_metric_callable = _default_format_reader

    result_paths = read_metric_results(path)

    metric_results: List[MetricResults] = []

    for result in result_paths:
        estimator_name, metric_name, split = name_metric_callable(result)
        result_df = pd.read_csv(result)
        _format_metric_results(
            [(metric_name, result_df)], estimator_name, metric_results, split == "train"
        )

    return metric_results
