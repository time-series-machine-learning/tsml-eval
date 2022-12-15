# -*- coding: utf-8 -*-
import platform

from tsml_eval.evaluation import evaluate_metric_results
from tsml_eval.evaluation.diagrams import scatter_diagram


def test_scatter_plot():
    """Test scatter plot."""

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
        "../../../../results/", custom_classification
    )
    scatter_diagram(classification_results, compare_estimators_to=["HC2"])
    joe = ""
