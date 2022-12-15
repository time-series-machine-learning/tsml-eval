# -*- coding: utf-8 -*-
import os

import pandas as pd

from tsml_eval.evaluation._result_evaluation import (
    CLUSTER_METRIC_CALLABLES,
    evaluate_metric_results,
    evaluate_raw_results,
)
from tsml_eval.evaluation._utils import (
    combine_two_summary_df,
    metric_result_to_summary,
    read_metric_results,
)
from tsml_eval.evaluation.diagrams import critical_difference_diagram


def load_custom_data():
    import platform

    def _default_format_reader(path: str):
        if "Windows" in platform.platform():
            split_subdir = path.split("\\")
        else:
            split_subdir = path.split("/")
        metric_name = split_subdir[-1].split(".")[0]
        split_temp = split_subdir[-1].split("_")
        split = "train"
        if "TEST" in split_temp[0]:
            split = "test"
        estimator_name = split_subdir[-2]
        return estimator_name, metric_name, split

    fix_format(
        "C:\\Users\\chris\\Documents\\Phd\\repos\\distance-based-time-series-clustering\\results",
        "C:\\Users\\chris\\Documents\\Phd\\result\\distance-paper",
        _default_format_reader,
    )


metric_name_cov = {
    "AMI": "AMI",
    "ARI": "ARI",
    "ACC": "ACC",
    "MI": "MI",
    "NMI": "NMI",
    "RI": "RI",
}


def fix_format(path: str, new_out_path: str, name_metric_callable):
    result_paths = read_metric_results(path)

    result_dfs = {}

    for result in result_paths:
        estimator_name, metric_name, split = name_metric_callable(result)
        result_df = pd.read_csv(result)
        dataset_column = result_df.columns[0]
        cols = result_df.columns[1:]
        for col in cols:
            if col not in result_dfs:
                result_dfs[col] = pd.DataFrame()
            curr_df = result_df[[dataset_column, col]]
            curr_df.columns = ["folds", "0"]
            for key in metric_name_cov:
                if key in metric_name:
                    metric_name = metric_name_cov[key]
                    break

            path = f"{new_out_path}\\{estimator_name}_mean\\{estimator_name}-{col}-mean\\{split}"
            if not os.path.exists(path):
                os.makedirs(path)
            curr_df.to_csv(f"{path}\\{metric_name}.csv", index=False)


if __name__ == "__main__":
    """Evaluate clustering results."""
    # load_custom_data()

    raw = evaluate_raw_results(
        experiment_name="averaging_results",
        path="C:\\Users\\chris\\Documents\\Phd\\Result\\averaging-results",
        output_dir="out/",
        metrics=CLUSTER_METRIC_CALLABLES,
    )
    old_results = evaluate_metric_results(
        "C:\\Users\\chris\\Documents\\Phd\\result\\distance-paper"
    )
    summary_new_experiment = metric_result_to_summary(raw, split="test")
    summary_old_results = metric_result_to_summary(old_results, split="test")
    summary_new_experiment = summary_new_experiment.replace(
        {"kmeans-dtw": "kmeans-dba-dtw"}, regex=True
    )
    summary_new_experiment = summary_new_experiment.replace(
        {"kmeans-msm": "kmeans-dba-msm"}, regex=True
    )
    combined = combine_two_summary_df(summary_new_experiment, summary_old_results)
    combined = combined[~combined["estimator"].str.contains("kmedoids")]
    print(len(combined["dataset"].unique()))
    cd = critical_difference_diagram(combined)
    for fig in cd:
        fig.show()
