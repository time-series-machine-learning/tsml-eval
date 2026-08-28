"""Functions for running experiments using cross-validation."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "classification_cross_validation",
    "classification_cross_validation_folds",
    "regression_cross_validation",
    "regression_cross_validation_folds",
]

from numbers import Integral

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from tsml_eval.experiments.experiments import (
    run_classification_experiment,
    run_regression_experiment,
)
from tsml_eval.utils.experiments import _check_existing_results


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
    data_transforms=None,
    transform_train_only=False,
    overwrite=False,
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
    cv : int, splitter object or None, default=None
        Cross-validation strategy or number of folds. If None, up to 10-fold
        stratified cross-validation will be used.
    fold_ids : int, iterable of int or None, default=None
        Fold ids to run. If None, all folds will be run.
    classifier_name : str or None, default=None
        Name of classifier used in writing results. If None, the name is taken from
        the classifier.
    dataset_name : str, default="N/A"
        Name of dataset.
    build_test_file : bool, default=True:
        Whether to generate test files or not. If the classifier can generate its own
        train probabilities, the classifier will be built but no file will be output.
    build_train_file : bool, default=False
        Whether to generate a train-estimate file for each selected outer fold. This
        uses the classifier's own train estimates when available, otherwise an inner
        cross-validation on that outer fold's training partition.
    ignore_custom_train_estimate : bool, default=False
        todo
    attribute_file_path : str or None, default=None
        todo (only test)
    att_max_shape : int, default=0
        todo
    benchmark_time : bool, default=True
        Whether to benchmark the hardware for each selected fold using a simple
        function and write the result. This will typically take ~2 seconds per fold,
        but is hardware dependent.
    data_transforms : transformer, list of transformers or None, default=None
        Transformer(s) to apply independently within each cross-validation fold.
        Transformers are cloned before each fold to prevent fitted state leaking
        between folds.
    transform_train_only : bool, default=False
        If True, apply data transforms only to the training data in each fold.
    overwrite : bool, default=False
        If False, existing fold result files are skipped. If True, existing files are
        overwritten.
    """
    _validate_file_options(build_test_file, build_train_file)
    folds = classification_cross_validation_folds(X, y, cv=cv)
    fold_ids = _normalise_fold_ids(fold_ids, len(folds))

    if classifier_name is None:
        classifier_name = type(estimator).__name__

    for fold, (train, test) in enumerate(folds):
        if fold not in fold_ids:
            continue

        fold_build_test, fold_build_train = _check_fold_results(
            results_path,
            classifier_name,
            dataset_name,
            fold,
            overwrite,
            build_test_file,
            build_train_file,
        )
        if not fold_build_test and not fold_build_train:
            continue

        run_classification_experiment(
            _safe_index(X, train),
            _safe_index(y, train),
            _safe_index(X, test),
            _safe_index(y, test),
            clone(estimator),
            results_path,
            classifier_name=classifier_name,
            dataset_name=dataset_name,
            resample_id=fold,
            data_transforms=_clone_data_transforms(data_transforms),
            transform_train_only=transform_train_only,
            build_test_file=fold_build_test,
            build_train_file=fold_build_train,
            ignore_custom_train_estimate=ignore_custom_train_estimate,
            attribute_file_path=_fold_attribute_path(attribute_file_path, fold),
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
    cv : int, splitter object or None, default=None
        Cross-validation strategy or number of folds. If None, up to 10-fold
        stratified cross-validation will be used.
    """
    cv = _resolve_classification_cv(y, cv)
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
    data_transforms=None,
    overwrite=False,
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
    cv : int, splitter object or None, default=None
        Cross-validation strategy or number of folds. If None, 10-fold shuffled
        cross-validation will be used.
    fold_ids : int, iterable of int or None, default=None
        Fold ids to run. If None, all folds will be run.
    regressor_name : str or None, default=None
        Name of regressor used in writing results. If None, the name is taken from
        the regressor.
    dataset_name : str, default=""
        Name of dataset.
    build_test_file : bool, default=True:
        Whether to generate test files or not. If the regressor can generate its own
        train predictions, the classifier will be built but no file will be output.
    build_train_file : bool, default=False
        Whether to generate a train-estimate file for each selected outer fold. This
        uses the regressor's own train estimates when available, otherwise an inner
        cross-validation on that outer fold's training partition.
    ignore_custom_train_estimate : bool, default=False
        todo
    attribute_file_path : str or None, default=None
        todo (only test)
    att_max_shape : int, default=0
        todo
    benchmark_time : bool, default=True
        Whether to benchmark the hardware for each selected fold using a simple
        function and write the result. This will typically take ~2 seconds per fold,
        but is hardware dependent.
    data_transforms : transformer, list of transformers or None, default=None
        Transformer(s) to apply independently within each cross-validation fold.
        Transformers are cloned before each fold to prevent fitted state leaking
        between folds.
    overwrite : bool, default=False
        If False, existing fold result files are skipped. If True, existing files are
        overwritten.
    """
    _validate_file_options(build_test_file, build_train_file)
    folds = regression_cross_validation_folds(X, y, cv=cv)
    fold_ids = _normalise_fold_ids(fold_ids, len(folds))

    if regressor_name is None:
        regressor_name = type(estimator).__name__

    for fold, (train, test) in enumerate(folds):
        if fold not in fold_ids:
            continue

        fold_build_test, fold_build_train = _check_fold_results(
            results_path,
            regressor_name,
            dataset_name,
            fold,
            overwrite,
            build_test_file,
            build_train_file,
        )
        if not fold_build_test and not fold_build_train:
            continue

        run_regression_experiment(
            _safe_index(X, train),
            _safe_index(y, train),
            _safe_index(X, test),
            _safe_index(y, test),
            clone(estimator),
            results_path,
            regressor_name=regressor_name,
            dataset_name=dataset_name,
            resample_id=fold,
            data_transforms=_clone_data_transforms(data_transforms),
            build_test_file=fold_build_test,
            build_train_file=fold_build_train,
            ignore_custom_train_estimate=ignore_custom_train_estimate,
            attribute_file_path=_fold_attribute_path(attribute_file_path, fold),
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
    cv : int, splitter object or None, default=None
        Cross-validation strategy or number of folds. If None, 10-fold shuffled
        cross-validation will be used.
    """
    cv = _resolve_regression_cv(cv)
    return list(cv.split(X, y))


def _resolve_classification_cv(y, cv):
    """Create the default classification splitter or validate an integer input."""
    if cv is None:
        cv_size = 10
        _, counts = np.unique(y, return_counts=True)
        cv_size = min(cv_size, int(np.min(counts)))
        if cv_size < 2:
            raise ValueError(
                "All classes must have at least 2 values to run the default "
                "cross-validation."
            )
        return StratifiedKFold(n_splits=cv_size, shuffle=True, random_state=0)

    if isinstance(cv, Integral) and not isinstance(cv, bool):
        if cv < 2:
            raise ValueError("cv must be at least 2.")
        return StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=0)

    if not callable(getattr(cv, "split", None)):
        raise TypeError("cv must be an integer or an object with a split method.")
    return cv


def _resolve_regression_cv(cv):
    """Create the default regression splitter or validate an integer input."""
    if cv is None:
        return KFold(n_splits=10, shuffle=True, random_state=0)

    if isinstance(cv, Integral) and not isinstance(cv, bool):
        if cv < 2:
            raise ValueError("cv must be at least 2.")
        return KFold(n_splits=int(cv), shuffle=True, random_state=0)

    if not callable(getattr(cv, "split", None)):
        raise TypeError("cv must be an integer or an object with a split method.")
    return cv


def _normalise_fold_ids(fold_ids, n_folds):
    """Validate fold ids and return them as a set for membership checks."""
    if n_folds < 1:
        raise ValueError("Cross-validation must contain at least one fold.")
    if fold_ids is None:
        return set(range(n_folds))
    if isinstance(fold_ids, Integral) and not isinstance(fold_ids, bool):
        fold_ids = [int(fold_ids)]
    else:
        try:
            fold_ids = list(fold_ids)
        except TypeError as exc:
            raise TypeError(
                "fold_ids must be an integer or an iterable of integers."
            ) from exc

    if len(fold_ids) == 0:
        raise ValueError("fold_ids must contain at least one fold id.")
    if any(
        not isinstance(fold, Integral) or isinstance(fold, bool) for fold in fold_ids
    ):
        raise TypeError("fold_ids must contain only integers.")

    fold_ids = [int(fold) for fold in fold_ids]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("fold_ids must not contain duplicate fold ids.")
    if any(fold < 0 or fold >= n_folds for fold in fold_ids):
        raise ValueError(f"fold_ids must be between 0 and {n_folds - 1}.")
    return set(fold_ids)


def _safe_index(data, indices):
    """Index numpy, pandas, and list-based aeon data containers."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    if isinstance(data, (list, tuple)):
        indices = np.asarray(indices)
        if indices.dtype == bool:
            indices = np.flatnonzero(indices)
        return [data[int(index)] for index in indices]
    return data[indices]


def _clone_data_transforms(data_transforms):
    """Clone transforms so fitted state cannot leak between folds."""
    if data_transforms is None:
        return None
    if isinstance(data_transforms, list):
        return [clone(transform) for transform in data_transforms]
    return clone(data_transforms)


def _check_fold_results(
    results_path,
    estimator_name,
    dataset_name,
    fold,
    overwrite,
    build_test_file,
    build_train_file,
):
    """Apply the standard result-existence check to one CV fold."""
    result_dataset = (
        "" if dataset_name is None or dataset_name in ("", "N/A") else dataset_name
    )
    return _check_existing_results(
        results_path,
        estimator_name,
        result_dataset,
        fold,
        overwrite,
        build_test_file,
        build_train_file,
    )


def _fold_attribute_path(attribute_file_path, fold):
    """Return a fold-specific attribute path."""
    if attribute_file_path is None:
        return None
    return f"{attribute_file_path}/fold{fold}/"


def _validate_file_options(build_test_file, build_train_file):
    """Require at least one result file for a CV experiment."""
    if not build_test_file and not build_train_file:
        raise ValueError(
            "Both test_file and train_file are set to False. At least one must be "
            "written."
        )
