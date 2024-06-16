# API

This page contains the auto-generated API documentation for `tsml-eval` package
functions and classes.

## Estimators: [tsml_eval.estimators](https://github.com/time-series-machine-learning/tsml-eval/tree/main/tsml_eval/estimators)

Wrappers and misc estimators for use with `tsml-eval`.

```{eval-rst}
.. currentmodule:: tsml_eval
.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    estimators.SklearnToTsmlClassifier
    estimators.SklearnToTsmlClusterer
    estimators.SklearnToTsmlRegressor
    estimators.classification.hybrid.FromFileHIVECOTE
    estimators.clustering.consensus.IterativeVotingClustering
    estimators.clustering.consensus.FromFileIterativeVotingClustering
    estimators.clustering.consensus.SimpleVote
    estimators.clustering.consensus.FromFileSimpleVote
```

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
    experiments.run_clustering_experiment
    experiments.load_and_run_clustering_experiment
    experiments.run_forecasting_experiment
    experiments.load_and_run_forecasting_experiment
    experiments.run_regression_experiment
    experiments.load_and_run_regression_experiment
    experiments.get_classifier_by_name
    experiments.get_clusterer_by_name
    experiments.get_forecaster_by_name
    experiments.get_regressor_by_name
    experiments.classification_cross_validation
    experiments.classification_cross_validation_folds
    experiments.regression_cross_validation
    experiments.regression_cross_validation_folds
```

## Utilities: [tsml_eval.utils](https://github.com/time-series-machine-learning/tsml-eval/tree/main/tsml_eval/utils)

Public utility functions used elsewhere in the package.

```{eval-rst}
.. currentmodule:: tsml_eval
.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    utils.arguments.parse_args
    utils.datasets.load_experiment_data
    utils.datasets.copy_dataset_ts_files
    utils.datasets.save_merged_dataset_splits
    utils.estimator_validation.is_sklearn_estimator
    utils.estimator_validation.is_sklearn_classifier
    utils.estimator_validation.is_sklearn_regressor
    utils.estimator_validation.is_sklearn_clusterer
    utils.experiments.assign_gpu
    utils.experiments.timing_benchmark
    utils.experiments.estimator_attributes_to_file
    utils.functions.str_in_nested_list
    utils.functions.pair_list_to_dict
    utils.functions.time_to_milliseconds
    utils.functions.rank_array
    utils.memory_recorder.record_max_memory
    utils.publications.extract_publication_csv_from_evaluation
    utils.publications.parameter_table_from_estimator_selection_function
    utils.resampling.resample_data
    utils.resampling.resample_data_indices
    utils.resampling.stratified_resample_data
    utils.resampling.stratified_resample_data_indices
    utils.results_repair.fix_broken_second_line
    utils.results_validation.validate_results_file
    utils.results_validation.compare_result_file_resample
    utils.results_writing.write_classification_results
    utils.results_writing.write_regression_results
    utils.results_writing.write_clustering_results
    utils.results_writing.write_forecasting_results
    utils.results_writing.write_results_to_tsml_format
```
