"""Functions for running experiments using cross-validation."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "classification_cross_validation",
    "classification_cross_validation_folds",
    "regression_cross_validation",
    "regression_cross_validation_folds",
]

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from tsml_eval.experiments.experiments import (
    run_classification_experiment,
    run_regression_experiment,
)


def classification_cross_validation(
    X,
    y,
    estimator,
    results_path,
    cv=None,
    fold_ids=None,
    classifier_name=None,
    dataset_name="N/A",
    build_test_file=True,
    build_train_file=False,
    ignore_custom_train_estimate=False,
    attribute_file_path=None,
    att_max_shape=0,
    benchmark_time=True,
):
    """Run a classification experiment using cross-validation.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target labels.
    estimator : object
        The classifier to be evaluated.
    results_path : str
        Path to save results.
    cv : object, optional
        Cross-validation strategy. If None, 10-fold cross-validation will be used.
    fold_ids : list, optional
        List of fold ids to run. If None, all folds will be run.
        row_normalise : bool, default=False
        Whether to normalise the data rows (time series) prior to fitting and
        predicting.
    classifier_name : str or None, default=None
        Name of classifier used in writing results. If None, the name is taken from
        the classifier.
    dataset_name : str, default="N/A"
        Name of dataset.
    build_test_file : bool, default=True:
        Whether to generate test files or not. If the classifier can generate its own
        train probabilities, the classifier will be built but no file will be output.
    build_train_file : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the classifier can produce its
        own estimates, those are used instead.
    ignore_custom_train_estimate : bool, default=False
        todo
    attribute_file_path : str or None, default=None
        todo (only test)
    att_max_shape : int, default=0
        todo
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    """
    if cv is None:
        cv_size = 10
        _, counts = np.unique(y, return_counts=True)
        min_class = np.min(counts)
        if min_class < cv_size:
            cv_size = min_class
            if cv_size < 2:
                raise ValueError(
                    "All classes must have at least 2 values to run the "
                    "default cross-validation."
                )
        cv = StratifiedKFold(n_splits=cv_size, shuffle=True, random_state=0)
    folds = list(cv.split(X, y))

    if fold_ids is not None:
        if isinstance(fold_ids, int):
            fold_ids = [fold_ids]
        if len(fold_ids) > len(folds) or max(fold_ids) >= len(folds):
            raise ValueError("Fold ids are to large for the number of folds.")
    else:
        fold_ids = list(range(len(folds)))

    for fold, (train, test) in enumerate(folds):
        if fold not in fold_ids:
            continue

        run_classification_experiment(
            X[train],
            y[train],
            X[test],
            y[test],
            estimator,
            results_path,
            classifier_name=classifier_name,
            dataset_name=dataset_name,
            resample_id=fold,
            build_test_file=build_test_file,
            build_train_file=build_train_file,
            ignore_custom_train_estimate=ignore_custom_train_estimate,
            attribute_file_path=attribute_file_path,
            att_max_shape=att_max_shape,
            benchmark_time=benchmark_time,
        )


def classification_cross_validation_folds(X, y, cv=None):
    """Get the folds for a classification cross-validation experiment.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target labels.
    cv : object, optional
        Cross-validation strategy. If None, 10-fold cross-validation will be used.
    """
    if cv is None:
        cv_size = 10
        _, counts = np.unique(y, return_counts=True)
        min_class = np.min(counts)
        if min_class < cv_size:
            cv_size = min_class
            if cv_size < 2:
                raise ValueError(
                    "All classes must have at least 2 values to run the "
                    "default cross-validation."
                )
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    return list(cv.split(X, y))


def regression_cross_validation(
    X,
    y,
    estimator,
    results_path,
    cv=None,
    fold_ids=None,
    regressor_name=None,
    dataset_name="",
    build_test_file=True,
    build_train_file=False,
    ignore_custom_train_estimate=False,
    attribute_file_path=None,
    att_max_shape=0,
    benchmark_time=True,
):
    """Run a regression experiment using cross-validation.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target labels.
    estimator : object
        The regressor to be evaluated.
    results_path : str
        Path to save results.
    cv : object, optional
        Cross-validation strategy. If None, 10-fold cross-validation will be used.
    fold_ids : list, optional
        List of fold ids to run. If None, all folds will be run.
        row_normalise : bool, default=False
        Whether to normalise the data rows (time series) prior to fitting and
        predicting.
    regressor_name : str or None, default=None
        Name of regressor used in writing results. If None, the name is taken from
        the regressor.
    dataset_name : str, default="N/A"
        Name of dataset.
    build_test_file : bool, default=True:
        Whether to generate test files or not. If the regressor can generate its own
        train predictions, the classifier will be built but no file will be output.
    build_train_file : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the regressor can produce its
        own estimates, those are used instead.
    ignore_custom_train_estimate : bool, default=False
        todo
    attribute_file_path : str or None, default=None
        todo (only test)
    att_max_shape : int, default=0
        todo
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    """
    if cv is None:
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
    folds = list(cv.split(X, y))

    if fold_ids is not None:
        if isinstance(fold_ids, int):
            fold_ids = [fold_ids]
        if len(fold_ids) > len(folds) or max(fold_ids) >= len(folds):
            raise ValueError("Fold ids are to large for the number of folds.")
    else:
        fold_ids = list(range(len(folds)))

    for fold, (train, test) in enumerate(folds):
        if fold not in fold_ids:
            continue

        run_regression_experiment(
            X[train],
            y[train],
            X[test],
            y[test],
            estimator,
            results_path,
            regressor_name=regressor_name,
            dataset_name=dataset_name,
            resample_id=fold,
            build_test_file=build_test_file,
            build_train_file=build_train_file,
            ignore_custom_train_estimate=ignore_custom_train_estimate,
            attribute_file_path=attribute_file_path,
            att_max_shape=att_max_shape,
            benchmark_time=benchmark_time,
        )


def regression_cross_validation_folds(X, y, cv=None):
    """Get the folds for a regression cross-validation experiment.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target labels.
    cv : object, optional
        Cross-validation strategy. If None, 10-fold cross-validation will be used.
    """
    if cv is None:
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
    return list(cv.split(X, y))
