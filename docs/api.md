# API

This page contains the auto-generated API documentation for tsml-eval package functions
and classes.

## Evaluation: [tsml_eval.evaluation](https://github.com/time-series-machine-learning/tsml-eval/tree/main/tsml_eval/evaluation)

Functions for evaluating the performance of a model.

```{eval-rst}
.. currentmodule:: tsml_eval
.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    evaluation.metrics.clustering_accuracy
    evaluation.metrics.davies_bouldin_score_from_file
```

## Experiments: [tsml_eval.experiments](https://github.com/time-series-machine-learning/tsml-eval/tree/main/tsml_eval/experiments)

Functions for running experiments.

```{eval-rst}
.. currentmodule:: tsml_eval
.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    experiments.run_classification_experiment
    experiments.load_and_run_classification_experiment
    experiments.run_regression_experiment
    experiments.load_and_run_regression_experiment
    experiments.run_clustering_experiment
    experiments.load_and_run_clustering_experiment
```

## Utilities: [tsml_eval.utils](https://github.com/time-series-machine-learning/tsml-eval/tree/main/tsml_eval/utils)

Public utility functions used elsewhere in the package.

```{eval-rst}
.. currentmodule:: tsml_eval
.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    utils.functions.str_in_nested_list
    utils.experiments.resample_data
    utils.experiments.stratified_resample_data
    utils.experiments.write_classification_results
    utils.experiments.write_regression_results
    utils.experiments.write_clustering_results
    utils.experiments.write_results_to_tsml_format
    utils.experiments.validate_results_file
    utils.experiments.fix_broken_second_line
    utils.experiments.compare_result_file_resample
    utils.experiments.assign_gpu
```
