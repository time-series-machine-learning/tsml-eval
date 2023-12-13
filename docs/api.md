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

    evaluation.evaluate_classifiers
    evaluation.evaluate_classifiers_from_file
    evaluation.evaluate_classifiers_by_problem
    evaluation.evaluate_clusterers
    evaluation.evaluate_clusterers_from_file
    evaluation.evaluate_clusterers_by_problem
    evaluation.evaluate_regressors
    evaluation.evaluate_regressors_from_file
    evaluation.evaluate_regressors_by_problem
    evaluation.evaluate_forecasters
    evaluation.evaluate_forecasters_from_file
    evaluation.evaluate_forecasters_by_problem
    evaluation.storage.ClassifierResults
    evaluation.storage.ClustererResults
    evaluation.storage.ForecasterResults
    evaluation.storage.RegressorResults
    evaluation.storage.load_classifier_results
    evaluation.storage.load_clusterer_results
    evaluation.storage.load_forecaster_results
    evaluation.storage.load_regressor_results
    evaluation.efficiency_benchmark.compare_estimators
    evaluation.efficiency_benchmark.benchmark_estimator
    evaluation.metrics.clustering_accuracy_score
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
    experiments.run_forecasting_experiment
    experiments.load_and_run_forecasting_experiment
```

## Utilities: [tsml_eval.utils](https://github.com/time-series-machine-learning/tsml-eval/tree/main/tsml_eval/utils)

Public utility functions used elsewhere in the package.

```{eval-rst}
.. currentmodule:: tsml_eval
.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    utils.arguments.parse_args
    utils.experiments.resample_data
    utils.experiments.stratified_resample_data
    utils.experiments.write_classification_results
    utils.experiments.write_regression_results
    utils.experiments.write_clustering_results
    utils.experiments.write_results_to_tsml_format
    utils.experiments.fix_broken_second_line
    utils.experiments.compare_result_file_resample
    utils.experiments.assign_gpu
    utils.experiments.timing_benchmark
    utils.experiments.estimator_attributes_to_file
    utils.functions.str_in_nested_list
    utils.functions.pair_list_to_dict
    utils.functions.time_to_milliseconds
    utils.functions.rank_array
    utils.memory_recorder.record_max_memory
    utils.validation.is_sklearn_estimator
    utils.validation.is_sklearn_classifier
    utils.validation.is_sklearn_regressor
    utils.validation.is_sklearn_clusterer
    utils.validation.validate_results_file
```
