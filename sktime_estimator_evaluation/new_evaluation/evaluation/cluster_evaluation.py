from typing import List, Union, Callable, Tuple
import os

import pandas as pd
from sktime_estimator_evaluation.new_evaluation.evaluation.utils import (
    resolve_experiment_paths,
    Experiment,
    MetricCallable,
    extract_estimator_experiment
)
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
)

CLUSTER_METRIC_CALLABLES = [
    MetricCallable(name="RI", callable=rand_score),
    MetricCallable(name="AMI", callable=adjusted_mutual_info_score),
    MetricCallable(name="NMI", callable=normalized_mutual_info_score),
    MetricCallable(name="ARI", callable=adjusted_rand_score),
    MetricCallable(name="MI", callable=mutual_info_score),
    MetricCallable(name="ACC", callable=accuracy_score),
]


def _get_metric_callable(
        metric: Union[str, MetricCallable],
        known_metrics: List[MetricCallable]
) -> MetricCallable:
    """Get the metric callable for the given metric.

    Parameters
    ----------
    metric: Union[str, Callable]
        Metric to get the callable for. If a string is provided, it must be a valid
        metric found in known_metrics.
    known_metrics: List[MetricCallable]
        List of known metrics.

    Returns
    -------
    MetricCallable
        Metric callable for the given metric.
    """

    if isinstance(metric, str):
        for curr_metric in CLUSTER_METRIC_CALLABLES:
            if curr_metric.name == metric:
                return curr_metric
        raise ValueError(f"Metric {metric} not found.")
    elif 'name' in metric and 'callable' in metric:
        return MetricCallable(**metric)
    else:
        raise ValueError(f"Metric {metric} not found. It must either be a metric"
                         f"callable (for example: "
                         f"MetricCallable(name='ARI', callable=adjusted_rand_score)"
                         f"or a string that is valid metric name. You can import "
                         f"MetricCallable like so: "
                         f"from "
                         f"sktime_estimator_evaluation.new_evaluation.evaluation.utils "
                         f"import MetricCallable")


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


def evaluate_cluster_results(
        experiment_name: str,
        path: str,
        metrics: List[Union[MetricCallable, str]] = None,
        output_dir: str = None
) -> pd.DataFrame:
    """Evaluate the results of a clustering experiment.

    Parameters
    ----------
    experiment_name: str
        Name of the experiment.
    path: str
        Path to a directory containing the results of a clustering experiment. This
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
    metrics: List[Union[MetricCallable, str]], defaults = CLUSTER_METRIC_CALLABLES
        List of metrics to evaluate. If a string is provided, it must be a valid metric
        found in CLUSTER_METRIC_CALLABLES.
    output_dir: str
        Path to directory to output result. If it doesn't exist then it is created

    Returns
    -------
    pd.DataFrame
        Dataframe containing the results of the evaluation. It will take the form:
        | Dataset  | Estimator 1 | Estimator 2 | Estimator 3 | ... |
        -----------------------------------------------------------
        | dataset1 | 0.0         | 0.0          | 0.0        | ... |
        | dataset2 | 0.0         | 0.0          | 0.0        | ... |
        | dataset3 | 0.0         | 0.0          | 0.0        | ... |
        | dataset4 | 0.0         | 0.0          | 0.0        | ... |
    """
    if metrics is None:
        metrics = CLUSTER_METRIC_CALLABLES

    output_dir = _resolve_output_dir(output_dir)

    metrics = [
        _get_metric_callable(metric, CLUSTER_METRIC_CALLABLES) for metric in metrics
    ]
    result: Experiment = resolve_experiment_paths(path, experiment_name)

    def resolve_join_path(curr_path: str, join_to: str) -> Union[str]:
        temp_path = None
        if output_dir is not None:
            temp_path = _resolve_output_dir(os.path.join(curr_path, join_to))
        return temp_path

    def write_csv_metric(
            curr_path: str,
            split: str,
            metric_results: List[Tuple[str, pd.DataFrame]]
    ):
        path = resolve_join_path(curr_path, split)
        for i in range(0, len(metric_results)):
            metric_name, metric_result = metric_results[i]
            metric_result.to_csv(
                os.path.join(path, f'{metric_name}.csv'),
                index=False
            )

    estimator_result = {}
    # For each estimator
    test = result['estimators']
    first = test[0]
    for estimator in result['estimators']:
        estimator_output_path = resolve_join_path(output_dir, estimator['estimator_name'])
        estimator_result[estimator['estimator_name']] = {}

        # For each experiment for estimator (i.e. hyperparams)
        for experiment in estimator['experiment_results']:
            estimator_result[experiment['experiment_name']] = {}
            experiment_output_path = resolve_join_path(estimator_output_path, experiment['experiment_name'])

            train_metrics, test_metrics = extract_estimator_experiment(
                experiment, metrics
            )

            if estimator_output_path is not None:
                write_csv_metric(experiment_output_path, 'train', train_metrics)
                write_csv_metric(experiment_output_path, 'test', test_metrics)

            # TODO: Define a return format that will also be used for reading in





