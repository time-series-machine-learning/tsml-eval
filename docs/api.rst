.. _api:

API
===

This page contains the auto-generated API documnetation for tsml-eval package functions
and classes.

Evaluation: tsml_eval.evaluation
--------------------------------

Functions for evaluating the performance of a model.

.. currentmodule:: tsml_eval.evaluation

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    metrics.clustering_accuracy
    metrics.davies_bouldin_score_from_file

Experiments: tsml_eval.experiments
----------------------------------

Functions for running experiments.

.. currentmodule:: tsml_eval.experiments

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    run_classification_experiment
    load_and_run_classification_experiment
    run_regression_experiment
    load_and_run_regression_experiment
    run_clustering_experiment
    load_and_run_clustering_experiment

Utilities: tsml_eval.utils
--------------------------

Public utility functions used elsewhere in the package.

.. currentmodule:: tsml_eval.utils

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    experiments.resample_data
    experiments.stratified_resample_data
    experiments.write_classification_results
    experiments.write_regression_results
    experiments.write_clustering_results
    experiments.write_results_to_tsml_format
    experiments.validate_results_file
    experiments.fix_broken_second_line
    experiments.compare_result_file_resample
