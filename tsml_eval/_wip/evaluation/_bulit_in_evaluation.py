# -*- coding: utf-8 -*-
import platform
from typing import List, Union

import numpy as np
import pandas as pd

from tsml_eval.evaluation._result_evaluation import evaluate_metric_results
from tsml_eval.evaluation._utils import MetricResults, metric_result_to_summary

ListOrString = Union[List[str], str]

PATH_TO_CLASSIFICATION_RESULTS = "../../../results/"


def _resolve_to_list(x: ListOrString) -> List[str]:
    if isinstance(x, str):
        return [x]
    return x


def fetch_classifier_metric(
    metrics: ListOrString = None,
    classifiers: ListOrString = None,
    datasets: ListOrString = None,
    folds=30,
    summary_format: bool = True,
    return_numpy: bool = False,
) -> Union[pd.DataFrame, List[pd.DataFrame], np.ndarray, List[np.ndarray]]:
    """Fetch the metric for a classifier over a dataset.

    Parameters
    ----------
    metric: str, defaults = None
        The metric to fetch. If None then all metrics are used.
    classifiers: str or list of str, defaults = None
        The classifier to fetch the metric for. If None then all classifiers used
    datasets: str or list of str, defaults = None
        The dataset to fetch the metric for. If None then all datasets used
    folds: int
        The number of folds to use for the evaluation. NOTE: folds are 0 indexing
        so if you ask for '6' youll get folds 0-5 (i.e. 6 folds).
    summary_format: bool, default=True
        If True, return a summary of the metric. If False, return dataset formatted
        df
    return_numpy: bool, default=False
        If True, return a numpy array. If False, return a pandas df.
    """

    def custom_classification(path: str):
        # Check os to determine split value
        if "Windows" in platform.platform():
            split_subdir = path.split("\\")
        else:
            split_subdir = path.split("/")
        metric_name = "ACC"
        file_name_split = split_subdir[-1].split("_")
        estimator_name = file_name_split[0]
        split = file_name_split[0].split("FOLDS")[0].lower()
        return estimator_name, metric_name, split

    classification_results = evaluate_metric_results(
        PATH_TO_CLASSIFICATION_RESULTS, custom_classification
    )

    summary_results = metric_result_to_summary(classification_results)

    if metrics is None:
        metrics = list(set(summary_results.columns[2:]))

    if datasets is None:
        datasets = list(summary_results["dataset"].unique())

    if classifiers is None:
        classifiers = list(summary_results["estimator"].unique())

    metrics = _resolve_to_list(metrics)
    datasets = _resolve_to_list(datasets)
    classifiers = _resolve_to_list(classifiers)

    temp = []
    for metric in metrics:
        curr_metric_result: MetricResults = {
            "metric_name": metric,
            "test_estimator_results": [],
            "train_estimator_results": [],
        }
        for result in classification_results:
            if result["metric_name"] == metric:
                for estimator_result in result["test_estimator_results"]:
                    for classifier in classifiers:
                        if classifier == estimator_result["estimator_name"]:
                            df = estimator_result["result"]
                            curr = df[df[df.columns[0]].isin(datasets)]
                            curr = curr.iloc[:, 0 : folds + 1]
                            curr_metric_result["test_estimator_results"].append(
                                {"estimator_name": classifier, "result": curr}
                            )
                            break

                break
        temp.append(curr_metric_result)

    result = metric_result_to_summary(temp)

    if summary_format is False:
        return_result = []

        for curr_metric in metrics:
            metric_index = result.columns.get_loc(curr_metric)
            columns = ["Problem"] + classifiers
            rows = []
            for dataset in datasets:
                rows.append([dataset] + ([None] * len(classifiers)))

            for i in range(len(classifiers)):
                estimator = classifiers[i]
                curr_df = result[result["estimator"] == estimator]
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

    if return_numpy is True:
        return result.to_numpy()

    return result


if __name__ == "__main__":
    metric = "ACC"
    datasets = ["Chinatown", "ItalyPowerDemand"]
    classifiers = ["HC2", "InceptionTime", "ROCKET"]
    res = fetch_classifier_metric("ACC", classifiers, datasets, 6)
    joe = ""
    # res_np = fetch_classifier_metric('ACC', classifiers, datasets, 6, return_numpy=True)
    # res_dataset = fetch_classifier_metric('ACC', classifiers, folds=6, summary_format=False)
    # res_dataset_np = fetch_classifier_metric('ACC', classifiers, datasets, 6, return_numpy=True, summary_format=False)
