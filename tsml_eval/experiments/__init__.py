"""Experiment functions."""

__all__ = [
    "run_classification_experiment",
    "load_and_run_classification_experiment",
    "run_clustering_experiment",
    "load_and_run_clustering_experiment",
    "run_forecasting_experiment",
    "load_and_run_forecasting_experiment",
    "run_regression_experiment",
    "load_and_run_regression_experiment",
    "get_classifier_by_name",
    "get_clusterer_by_name",
    "get_forecaster_by_name",
    "get_regressor_by_name",
    "classification_cross_validation",
    "classification_cross_validation_folds",
    "regression_cross_validation",
    "regression_cross_validation_folds",
]

from tsml_eval.experiments.cross_validation import (
    classification_cross_validation,
    classification_cross_validation_folds,
    regression_cross_validation,
    regression_cross_validation_folds,
)
from tsml_eval.experiments.experiments import (
    load_and_run_classification_experiment,
    load_and_run_clustering_experiment,
    load_and_run_forecasting_experiment,
    load_and_run_regression_experiment,
    run_classification_experiment,
    run_clustering_experiment,
    run_forecasting_experiment,
    run_regression_experiment,
)
from tsml_eval.experiments.set_classifier import get_classifier_by_name
from tsml_eval.experiments.set_clusterer import get_clusterer_by_name
from tsml_eval.experiments.set_forecaster import get_forecaster_by_name
from tsml_eval.experiments.set_regressor import get_regressor_by_name
