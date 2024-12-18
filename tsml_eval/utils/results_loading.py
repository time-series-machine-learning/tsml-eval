"""Utilities for results evaluation."""

__all__ = [
    "load_estimator_results",
    "estimator_results_to_dict",
    "load_estimator_results_to_dict",
    "estimator_results_to_array",
    "load_estimator_results_to_array",
]

import numpy as np
from aeon.benchmarking.results_loaders import (
    _results_dict_to_array as _aeon_results_dict_to_array,
)

from tsml_eval.evaluation.storage import (
    load_classifier_results,
    load_clusterer_results,
    load_forecaster_results,
    load_regressor_results,
)
from tsml_eval.utils.results_validation import (
    _check_classification_third_line,
    _check_clustering_third_line,
    _check_forecasting_third_line,
    _check_regression_third_line,
)


def load_estimator_results(file_path, calculate_stats=True, verify_values=True):
    """
    Load and return estimator results from a specified file.

    This function reads a file containing estimator results and reconstructs the
    EstimatorResults object. It optionally calculates performance statistics and
    verifies values based on the loaded data.

    Parameters
    ----------
    file_path : str
        The path to the file from which estimator results should be loaded. The file
        should be a tsml formatted estimator results file.
    calculate_stats : bool, default=True
        Whether to calculate performance statistics from the loaded results.
    verify_values : bool, default=True
        If the function should perform verification of the loaded values.

    Returns
    -------
    er : EstimatorResults
        A EstimatorResults object containing the results loaded from the file.
    """
    with open(file_path) as f:
        lines = [next(f) for _ in range(3)]

    if _check_classification_third_line(lines[2]):
        return load_classifier_results(
            file_path, calculate_stats=calculate_stats, verify_values=verify_values
        )
    elif _check_clustering_third_line(lines[2]):
        return load_clusterer_results(
            file_path, calculate_stats=calculate_stats, verify_values=verify_values
        )
    elif _check_regression_third_line(lines[2]):
        return load_regressor_results(
            file_path, calculate_stats=calculate_stats, verify_values=verify_values
        )
    elif _check_forecasting_third_line(lines[2]):
        return load_forecaster_results(
            file_path, calculate_stats=calculate_stats, verify_values=verify_values
        )
    else:
        raise ValueError("Unable to determine the type of results file.")


def estimator_results_to_dict(estimator_results, measure):
    """Convert a list of EstimatorResults objects to a dictionary of metrics.

    Follows the output format of ``aeon`` dict results loaders.

    Parameters
    ----------
    estimator_results : list of EstimatorResults
        The results to convert.
    measure : str
        A valid metric attribute from the relevant EstimatorResults object.
        Dependent on the task, i.e. for classification, "accuracy", "auroc", "balacc",
        and regression, "mse", "mae", "r2".

    Returns
    -------
    results: dict
        Dictionary with estimator name keys containing another dictionary.
        Sub-dictionary consists of dataset name keys and contains of scores for each
        dataset.
    """
    res_dict = _create_results_dictionary(estimator_results)

    resamples = None
    for estimator in res_dict:
        for dataset in res_dict[estimator]:
            if len(res_dict[estimator][dataset].keys()) != 1:
                raise ValueError(
                    "The split value for all results must be the same i.e. "
                    "no mixing of train and test results. Found: "
                    f"{res_dict[estimator][dataset].keys()} for {estimator}/{dataset}"
                )

            for split in res_dict[estimator][dataset]:
                values = []
                keys = list(res_dict[estimator][dataset][split].keys())
                if resamples is None:
                    resamples = keys
                else:
                    if resamples != keys:
                        raise ValueError(
                            "The resample_id values for each estimator/dataset "
                            f"combination must be the same. Expected: {resamples}, "
                            f"found: {keys} for {estimator}/{dataset}"
                        )

                for resample in res_dict[estimator][dataset][split]:
                    res = res_dict[estimator][dataset][split][resample]
                    res.calculate_statistics()
                    values.append(getattr(res, measure))

                values = np.array(values) if len(values) > 1 else values[0]
                res_dict[estimator][dataset] = values

    return res_dict


def load_estimator_results_to_dict(
    load_path, estimator_names, dataset_names, measure, resamples=None, split="test"
):
    """Load and convert EstimatorResults objects to a dictionary of metrics.

    Follows the output format of ``aeon`` dict results loaders.

    Parameters
    ----------
    load_path : str or list of str
        The path to the collection of estimator result files to convert.
        If load_path is a list, it will load results from each path in the list. It
        is expected that estimator_names and dataset_names are lists of lists with
        the same length as load_path.
    estimator_names : list of str, or list of list
        The names of the estimators to convert.
        If load_path is a list, estimator_names must be a list of lists with the same
        length as load_path.
    dataset_names : str, list of str or list of list
        The names of the datasets to convert. If a list of strings, each item is the
        name of a dataset. If a string, it is the path to a file containing the names
        of the datasets, one per line.
        If load_path is a list, dataset_names must be a list of lists with the same
        length as load_path.
    measure : str
        A valid metric attribute from the relevant EstimatorResults object.
        Dependent on the task, i.e. for classification, "accuracy", "auroc", "balacc",
        and regression, "mse", "mae", "r2".
    resamples : int or list of int, default=None
        The resamples to load. If int, loads resamples 0 to resamples-1.
        For 1 or None, only the loaded score is returned.
        For 2 or more, a np.ndarray of scores for all resamples up to resamples are
        returned.
    split : str, default="test"
        The split to load results for, appears at the start of the results file name.
        Should be one of "train", "test" in most circumstances.

    Returns
    -------
    results: dict
        Dictionary with estimator name keys containing another dictionary.
        Sub-dictionary consists of dataset name keys and contains of scores for each
        dataset.
    """
    load_path, estimator_names, dataset_names, resamples = _load_by_problem_init(
        "estimator", load_path, estimator_names, dataset_names, resamples
    )

    estimator_results = []
    for i, path in enumerate(load_path):
        for estimator_name in estimator_names[i]:
            for dataset_name in dataset_names[i]:
                for resample in resamples:
                    result = load_estimator_results(
                        f"{path}/{estimator_name}/Predictions/"
                        f"{dataset_name}/{split}Resample{resample}.csv",
                    )
                    estimator_results.append(result)

    return estimator_results_to_dict(estimator_results, measure)


def estimator_results_to_array(estimator_results, measure, include_missing=False):
    """Convert a list of EstimatorResults objects to an array of metrics.

    Follows the output format of ``aeon`` array results loaders.

    Parameters
    ----------
    estimator_results : list of EstimatorResults
        The results to convert.
    measure : str
        A valid metric attribute from the relevant EstimatorResults object.
        Dependent on the task, i.e. for classification, "accuracy", "auroc", "balacc",
        and regression, "mse", "mae", "r2".
    include_missing : bool, default=False
        Whether to include datasets with missing results in the output.
        If False, the whole problem is ignored if any estimator is missing results it.
        If True, NaN is returned instead of a score in missing cases.

    Returns
    -------
    results: 2D numpy array
        Array of scores. Each column is a results for an estimator, each row a dataset.
        For multiple resamples, these scores are averaged.
    dataset_names: list of str
        List of dataset names that were retained.
    estimator_names: list of str
        List of estimator names that were retained.
    """
    res_dict = estimator_results_to_dict(estimator_results, measure)
    return _results_dict_to_array(res_dict, None, None, include_missing)


def load_estimator_results_to_array(
    load_path, estimator_names, dataset_names, measure, resamples=None, split="test"
):
    """Load and convert EstimatorResults objects to an array of metrics.

    Follows the output format of ``aeon`` array results loaders.

    Parameters
    ----------
    load_path : str or list of str
        The path to the collection of estimator result files to convert.
        If load_path is a list, it will load results from each path in the list. It
        is expected that estimator_names and dataset_names are lists of lists with
        the same length as load_path.
    estimator_names : list of str, or list of list
        The names of the estimators to convert.
        If load_path is a list, estimator_names must be a list of lists with the same
        length as load_path.
    dataset_names : str, list of str or list of list
        The names of the datasets to convert. If a list of strings, each item is the
        name of a dataset. If a string, it is the path to a file containing the names
        of the datasets, one per line.
        If load_path is a list, dataset_names must be a list of lists with the same
        length as load_path.
    measure : str
        A valid metric attribute from the relevant EstimatorResults object.
        Dependent on the task, i.e. for classification, "accuracy", "auroc", "balacc",
        and regression, "mse", "mae", "r2".
    resamples : int or list of int, default=None
        The resamples to load. If int, loads resamples 0 to resamples-1.
        For 1 or None, only the loaded score is returned.
        For 2 or more, the scores of all resamples up to resamples are averaged and
        returned.
    split : str, default="test"
        The split to load results for, appears at the start of the results file name.
        Should be one of "train", "test" in most circumstances.

    Returns
    -------
    results: 2D numpy array
        Array of scores. Each column is a results for an estimator, each row a dataset.
        For multiple resamples, these scores are averaged.
    dataset_names: list of str
        List of dataset names that were retained.
    estimator_names: list of str
        List of estimator names that were retained.
    """
    res_dict = load_estimator_results_to_dict(
        load_path,
        estimator_names,
        dataset_names,
        measure,
        resamples=resamples,
        split=split,
    )
    return _results_dict_to_array(res_dict, estimator_names, dataset_names, False)


def _results_dict_to_array(results_dict, estimators, datasets, include_missing):
    if estimators is None:
        estimators = list(results_dict.keys())
    if datasets is None:
        datasets = []
        for est in results_dict:
            datasets.extend(results_dict[est].keys())
        datasets = set(datasets)

    arr, datasets = _aeon_results_dict_to_array(
        results_dict, estimators, datasets, include_missing=include_missing
    )
    return arr, datasets, estimators


def _load_by_problem_init(type, load_path, estimator_names, dataset_names, resamples):
    if isinstance(load_path, str):
        load_path = [load_path]
    elif not isinstance(load_path, list):
        raise TypeError("load_path must be a str or list of str.")

    if isinstance(estimator_names[0], (str, tuple)):
        estimator_names = [estimator_names]
    elif not isinstance(estimator_names[0], list):
        raise TypeError(f"{type}_names must be a str, tuple or list of str or tuple.")

    if isinstance(dataset_names, str):
        with open(dataset_names) as f:
            dataset_names = f.readlines()
            dataset_names = [[d.strip() for d in dataset_names]] * len(load_path)
    elif isinstance(dataset_names[0], str):
        dataset_names = [dataset_names] * len(load_path)
    elif not isinstance(dataset_names[0], list):
        raise TypeError("dataset_names must be a str or list of str.")

    if len(load_path) != len(estimator_names) or len(load_path) != len(dataset_names):
        raise ValueError(
            f"load_path, {type}_names and dataset_names must be the same length."
        )

    if resamples is None:
        resamples = [""]
    elif isinstance(resamples, int):
        resamples = [str(i) for i in range(resamples)]
    else:
        resamples = [str(resample) for resample in resamples]

    return load_path, estimator_names, dataset_names, resamples


def _create_results_dictionary(estimator_results, estimator_names=None):
    results_dict = {}

    for i, estimator_result in enumerate(estimator_results):
        name = (
            estimator_result.estimator_name
            if estimator_names is None
            else estimator_names[i]
        )

        if results_dict.get(name) is None:
            results_dict[name] = {}

        if results_dict[name].get(estimator_result.dataset_name) is None:
            results_dict[name][estimator_result.dataset_name] = {}

        if (
            results_dict[name][estimator_result.dataset_name].get(
                estimator_result.split.lower()
            )
            is None
        ):
            results_dict[name][estimator_result.dataset_name][
                estimator_result.split.lower()
            ] = {}

        results_dict[name][estimator_result.dataset_name][
            estimator_result.split.lower()
        ][estimator_result.resample_id] = estimator_result

    return results_dict
