# -*- coding: utf-8 -*-
"""Evaluation module for aeon estimators."""
__all__ = [
    "evaluate_raw_results",
    "evaluate_metric_results",
    "CLUSTER_METRIC_CALLABLES",
    "CLASSIFICATION_METRIC_CALLABLES",
    "read_results_from_uea_format",
    "read_clusterer_result_from_uea_format",
    "ExperimentResamples",
    "Dataset",
    "EstimatorExperiment",
    "Estimator",
    "Experiment",
    "MetricCallable",
    "MetricResults",
    "EstimatorMetricResults",
    "read_metric_results",
    "extract_estimator_experiment",
    "resolve_experiment_paths",
    "metric_result_to_summary",
    "from_metric_dataset_format_to_metric_summary",
    "from_metric_summary_to_dataset_format",
]

from tsml_eval._wip.evaluation._bulit_in_evaluation import fetch_classifier_metric
from tsml_eval._wip.evaluation._result_evaluation import (
    CLASSIFICATION_METRIC_CALLABLES,
    CLUSTER_METRIC_CALLABLES,
    evaluate_metric_results,
    evaluate_raw_results,
)
