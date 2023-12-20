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
    "set_classifier",
    "set_clusterer",
    "set_forecaster",
    "set_regressor",
]

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
from tsml_eval.experiments.set_classifier import set_classifier
from tsml_eval.experiments.set_clusterer import set_clusterer
from tsml_eval.experiments.set_forecaster import set_forecaster
from tsml_eval.experiments.set_regressor import set_regressor
