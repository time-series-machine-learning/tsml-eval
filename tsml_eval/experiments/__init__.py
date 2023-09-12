"""Experiment functions."""

__all__ = [
    "run_classification_experiment",
    "load_and_run_classification_experiment",
    "run_regression_experiment",
    "load_and_run_regression_experiment",
    "run_clustering_experiment",
    "load_and_run_clustering_experiment",
    "run_forecasting_experiment",
    "load_and_run_forecasting_experiment",
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
