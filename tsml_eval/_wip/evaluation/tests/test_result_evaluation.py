# -*- coding: utf-8 -*-
import platform

from tsml_eval.evaluation._result_evaluation import (
    CLUSTER_METRIC_CALLABLES,
    evaluate_metric_results,
    evaluate_raw_results,
)


def test_cluster_evaluation():
    """Evaluate clustering results."""
    result = evaluate_raw_results(
        experiment_name="test",
        path="C:\\Users\\chris\\Documents\\Phd\\Results\\kmeans\\kmeans_dba",
        output_dir="out/",
        metrics=CLUSTER_METRIC_CALLABLES,
    )
    joe = ""


def test_classification_evaluation():
    """Evaluate classification results."""
    pass


def test_read_evaluation_metric_results():
    """Read evaluation metric results."""
    clustering_results = evaluate_metric_results("./out/")

    def custom_classification(path: str):
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
        "../../../results/", custom_classification
    )

    from tsml_eval._wip.evaluation.diagrams._critical_difference_diagram import (
        critical_difference_diagram,
    )

    # create_critical_difference_diagram(classification_results)
    critical_difference_diagram(clustering_results)
    joe = ""
