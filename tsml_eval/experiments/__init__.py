"""Experiment functions."""

__all__ = [
    "run_classification_experiment",
    "load_and_run_classification_experiment",
    "run_clustering_experiment",
    "load_and_run_clustering_experiment",
    "run_regression_experiment",
    "load_and_run_regression_experiment",
    "get_classifier_by_name",
    "get_clusterer_by_name",
    "get_regressor_by_name",
    "get_data_transform_by_name",
    "run_timing_experiment",
    "classification_cross_validation",
    "classification_cross_validation_folds",
    "regression_cross_validation",
    "regression_cross_validation_folds",
]

from tsml_eval.experiments._get_classifier import get_classifier_by_name
from tsml_eval.experiments._get_clusterer import get_clusterer_by_name
from tsml_eval.experiments._get_data_transform import get_data_transform_by_name
from tsml_eval.experiments._get_regressor import get_regressor_by_name
from tsml_eval.experiments.cross_validation import (
    classification_cross_validation,
    classification_cross_validation_folds,
    regression_cross_validation,
    regression_cross_validation_folds,
)
from tsml_eval.experiments.experiments import (
    load_and_run_classification_experiment,
    load_and_run_clustering_experiment,
    load_and_run_regression_experiment,
    run_classification_experiment,
    run_clustering_experiment,
    run_regression_experiment,
)
from tsml_eval.experiments.scalability import run_timing_experiment
