.. _api:

API
===

Evaluation
----------

.. currentmodule:: tsml_eval.evaluation

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    metrics.clustering_accuracy

Experiments
-----------

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

Utilities
---------

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
