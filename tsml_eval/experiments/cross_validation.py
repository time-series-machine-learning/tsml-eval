"""Functions for running experiments using cross-validation."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "classification_cross_validation",
    "classification_cross_validation_folds",
    "regression_cross_validation",
    "regression_cross_validation_folds",
]

from collections.abc import Iterable, Mapping
from numbers import Integral

import numpy as np
import pandas as pd
from aeon.classification import BaseClassifier
from aeon.regression.base import BaseRegressor
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
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
    groups=None,
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
    cv : int, splitter object, iterable of splits or None, default=None
        Cross-validation strategy, number of folds, or an iterable of ``(train, test)``
        index splits. If None, up to 10-fold stratified cross-validation will be used.
    fold_ids : int, iterable of int or None, default=None
        Fold ids to run. If None, all folds will be run.
    classifier_name : str or None, default=None
        Name of classifier used in writing results. If None, the name is taken from
        the classifier.
    dataset_name : str, default="N/A"
        Name of dataset.
    build_test_file : bool, default=True
        Whether to generate test files or not. If the classifier can generate its own
        train probabilities, the classifier will be built but no file will be output.
    build_train_file : bool, default=False
        Whether to generate a train-estimate file for each selected outer fold. This
        uses the classifier's own train estimates when available, otherwise an inner
        cross-validation on that outer fold's training partition.
        Cannot be combined with external ``data_transforms``: the underlying runner
        fits these before the inner split, which can leak validation information.
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
    data_transforms : transformer, sequence of transformers or None, default=None
        Transformer(s) to apply independently within each cross-validation fold.
        Transformers are cloned before each fold to prevent fitted state leaking
        between folds.
    transform_train_only : bool, default=False
        If True, apply data transforms only to the training data in each fold.
    overwrite : bool, default=False
        If False, existing fold result files are skipped. If True, existing files are
        overwritten.
    groups : array-like or None, default=None
        Group labels for the cases used while splitting the dataset into train and
        test folds. Only used in conjunction with a ``cv`` splitter.
    """
    uses_custom_train_estimate = _validate_classifier(
        estimator, build_train_file, ignore_custom_train_estimate
    )
    data_transforms = _validate_file_options(
        build_test_file,
        build_train_file,
        data_transforms,
        transform_train_only=transform_train_only,
    )
    n_cases, y_arr = _validate_X_y(X, y)
    splitter, splits = _resolve_classification_cv(y_arr, cv)
    folds = _generate_folds(X, y, n_cases, splitter, splits, groups)
    fold_ids = _normalise_fold_ids(fold_ids, len(folds))
    _validate_classification_folds(
        folds,
        fold_ids,
        y_arr,
        require_inner_cv=build_train_file and not uses_custom_train_estimate,
    )

    if classifier_name is None:
        classifier_name = type(estimator).__name__

    for fold in sorted(fold_ids):
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

        train, test = folds[fold]
        run_classification_experiment(
            _safe_index(X, train),
            y_arr[train],
            _safe_index(X, test),
            y_arr[test],
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


def classification_cross_validation_folds(X, y, cv=None, groups=None):
    """Get the folds for a classification cross-validation experiment.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target labels.
    cv : int, splitter object, iterable of splits or None, default=None
        Cross-validation strategy, number of folds, or an iterable of ``(train, test)``
        index splits. If None, up to 10-fold stratified cross-validation will be used.
    groups : array-like or None, default=None
        Group labels for the cases used while splitting the dataset into train and
        test folds. Only used in conjunction with a ``cv`` splitter.

    Returns
    -------
    folds : list of tuple of (np.ndarray, np.ndarray)
        List of ``(train_indices, test_indices)`` pairs for each fold.
    """
    n_cases, y_arr = _validate_X_y(X, y)
    splitter, splits = _resolve_classification_cv(y_arr, cv)
    return _generate_folds(X, y, n_cases, splitter, splits, groups)


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
    groups=None,
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
    cv : int, splitter object, iterable of splits or None, default=None
        Cross-validation strategy, number of folds, or an iterable of ``(train, test)``
        index splits. If None, 10-fold shuffled cross-validation will be used.
    fold_ids : int, iterable of int or None, default=None
        Fold ids to run. If None, all folds will be run.
    regressor_name : str or None, default=None
        Name of regressor used in writing results. If None, the name is taken from
        the regressor.
    dataset_name : str, default=""
        Name of dataset.
    build_test_file : bool, default=True
        Whether to generate test files or not. If the regressor can generate its own
        train predictions, the regressor will be built but no file will be output.
    build_train_file : bool, default=False
        Whether to generate a train-estimate file for each selected outer fold. This
        uses the regressor's own train estimates when available, otherwise an inner
        cross-validation on that outer fold's training partition.
        Cannot be combined with external ``data_transforms``: the underlying runner
        fits these before the inner split, which can leak validation information.
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
    data_transforms : transformer, sequence of transformers or None, default=None
        Transformer(s) to apply independently within each cross-validation fold.
        Transformers are cloned before each fold to prevent fitted state leaking
        between folds.
    overwrite : bool, default=False
        If False, existing fold result files are skipped. If True, existing files are
        overwritten.
    groups : array-like or None, default=None
        Group labels for the cases used while splitting the dataset into train and
        test folds. Only used in conjunction with a ``cv`` splitter.
    """
    uses_custom_train_estimate = _validate_regressor(
        estimator, build_train_file, ignore_custom_train_estimate
    )
    data_transforms = _validate_file_options(
        build_test_file, build_train_file, data_transforms
    )
    n_cases, y_arr = _validate_X_y(X, y)
    splitter, splits = _resolve_regression_cv(cv)
    folds = _generate_folds(X, y, n_cases, splitter, splits, groups)
    fold_ids = _normalise_fold_ids(fold_ids, len(folds))
    _validate_regression_folds(
        folds,
        fold_ids,
        require_inner_cv=build_train_file and not uses_custom_train_estimate,
    )

    if regressor_name is None:
        regressor_name = type(estimator).__name__

    for fold in sorted(fold_ids):
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

        train, test = folds[fold]
        run_regression_experiment(
            _safe_index(X, train),
            y_arr[train],
            _safe_index(X, test),
            y_arr[test],
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


def regression_cross_validation_folds(X, y, cv=None, groups=None):
    """Get the folds for a regression cross-validation experiment.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target labels.
    cv : int, splitter object, iterable of splits or None, default=None
        Cross-validation strategy, number of folds, or an iterable of ``(train, test)``
        index splits. If None, 10-fold shuffled cross-validation will be used.
    groups : array-like or None, default=None
        Group labels for the cases used while splitting the dataset into train and
        test folds. Only used in conjunction with a ``cv`` splitter.

    Returns
    -------
    folds : list of tuple of (np.ndarray, np.ndarray)
        List of ``(train_indices, test_indices)`` pairs for each fold.
    """
    n_cases, _ = _validate_X_y(X, y)
    splitter, splits = _resolve_regression_cv(cv)
    return _generate_folds(X, y, n_cases, splitter, splits, groups)


def _is_pd_multiindex_collection(data):
    """Return True if data is a two-level MultiIndex pandas DataFrame collection."""
    return (
        isinstance(data, pd.DataFrame)
        and isinstance(data.index, pd.MultiIndex)
        and data.index.nlevels == 2
    )


def _validate_X_y(X, y):
    """Validate X and y case counts and return ``(n_cases, y_array)``."""
    n_cases = (
        len(X.index.get_level_values(0).unique())
        if _is_pd_multiindex_collection(X)
        else len(X)
    )
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if len(y_arr) != n_cases:
        raise ValueError(
            f"X and y must have the same number of cases, got {n_cases} and "
            f"{len(y_arr)}."
        )
    return n_cases, y_arr


def _validate_classifier(estimator, build_train_file, ignore_custom_train_estimate):
    """Validate a classifier and return whether it uses a custom train estimate."""
    if isinstance(estimator, BaseClassifier):
        return (
            build_train_file
            and not ignore_custom_train_estimate
            and bool(estimator.get_tag("capability:train_estimate", False, False))
        )
    if isinstance(estimator, BaseEstimator) and is_classifier(estimator):
        return False
    raise TypeError("classifier must be an aeon, tsml or sklearn classifier.")


def _validate_regressor(estimator, build_train_file, ignore_custom_train_estimate):
    """Validate a regressor and return whether it uses a custom train estimate."""
    if isinstance(estimator, BaseRegressor):
        return (
            build_train_file
            and not ignore_custom_train_estimate
            and bool(estimator.get_tag("capability:train_estimate", False, False))
        )
    if isinstance(estimator, BaseEstimator) and is_regressor(estimator):
        return False
    raise TypeError("regressor must be a tsml, aeon or sklearn regressor.")


def _resolve_classification_cv(y_arr, cv):
    """Create the default classification splitter or validate ``cv``."""
    classes, counts = np.unique(y_arr, return_counts=True)
    if cv is None and (len(counts) == 0 or int(np.min(counts)) < 2):
        raise ValueError(
            "All classes must have at least 2 values to run the default "
            "cross-validation."
        )
    if len(classes) < 2:
        raise ValueError(
            "y must contain at least 2 unique classes for classification "
            "cross-validation."
        )

    if cv is None:
        cv_size = min(10, int(np.min(counts)))
        return StratifiedKFold(n_splits=cv_size, shuffle=True, random_state=0), None

    if isinstance(cv, Integral) and not isinstance(cv, bool):
        if cv < 2:
            raise ValueError("cv must be at least 2.")
        return StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=0), None

    return _resolve_custom_cv(cv)


def _resolve_regression_cv(cv):
    """Create the default regression splitter or validate ``cv``."""
    if cv is None:
        return KFold(n_splits=10, shuffle=True, random_state=0), None

    if isinstance(cv, Integral) and not isinstance(cv, bool):
        if cv < 2:
            raise ValueError("cv must be at least 2.")
        return KFold(n_splits=int(cv), shuffle=True, random_state=0), None

    return _resolve_custom_cv(cv)


def _resolve_custom_cv(cv):
    """Validate a custom splitter object or iterable of pre-computed splits."""
    if not isinstance(cv, type) and callable(getattr(cv, "split", None)):
        return cv, None

    if isinstance(cv, Iterable) and not isinstance(
        cv, (str, bytes, bytearray, Mapping)
    ):
        return None, list(cv)

    raise TypeError(
        "cv must be an integer, an object with a split method, or an iterable of "
        "(train, test) splits."
    )


def _generate_folds(X, y, n_cases, splitter, splits, groups):
    """Materialise and validate ``(train, test)`` fold index pairs."""
    if splitter is not None:
        split_X = np.zeros(n_cases) if _is_pd_multiindex_collection(X) else X
        raw_splits = (
            splitter.split(split_X, y)
            if groups is None
            else splitter.split(split_X, y, groups)
        )
        splits = list(raw_splits)
    elif groups is not None:
        raise ValueError(
            "groups cannot be provided when cv is an iterable of pre-computed splits."
        )

    if len(splits) < 1:
        raise ValueError("Cross-validation must contain at least one fold.")

    in_train = np.zeros(n_cases, dtype=bool)
    validated = []
    for split in splits:
        if isinstance(split, (str, bytes, bytearray, Mapping)):
            raise ValueError(
                "Each cross-validation split must be a (train, test) pair of index "
                "arrays."
            )
        try:
            train_raw, test_raw = split
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Each cross-validation split must be a (train, test) pair of index "
                "arrays."
            ) from exc
        train = _normalise_split_indices(train_raw, n_cases, "train")
        test = _normalise_split_indices(test_raw, n_cases, "test")
        in_train[train] = True
        has_overlap = bool(np.any(in_train[test]))
        in_train[train] = False
        if has_overlap:
            raise ValueError(
                "Cross-validation train and test indices must be disjoint within "
                "each fold."
            )
        validated.append((train, test))
    return validated


def _normalise_split_indices(indices, n_cases, split_name):
    """Validate a fold's train or test indices and return a 1D integer array."""
    if isinstance(indices, (str, bytes, bytearray, Mapping)):
        raise TypeError(
            f"{split_name} indices must be an array-like of integers or booleans."
        )
    arr = np.asarray(indices)
    if arr.ndim != 1:
        raise ValueError(f"{split_name} indices must be one-dimensional.")
    if arr.dtype == bool:
        if len(arr) != n_cases:
            raise ValueError(
                f"Boolean {split_name} mask length ({len(arr)}) must match the "
                f"number of cases ({n_cases})."
            )
        arr = np.flatnonzero(arr)
    elif arr.size == 0:
        arr = np.empty(0, dtype=int)
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(int, copy=False)
        if np.any(arr < 0) or np.any(arr >= n_cases):
            raise ValueError(
                f"{split_name} indices must be between 0 and {n_cases - 1}."
            )
    else:
        raise TypeError(f"{split_name} indices must contain integers or booleans.")

    if arr.size == 0:
        raise ValueError(
            f"Each cross-validation fold must contain at least one {split_name} case."
        )
    return arr


def _validate_classification_folds(folds, fold_ids, y_arr, require_inner_cv):
    """Ensure selected folds preserve full class coverage and inner-CV counts."""
    all_classes = np.unique(y_arr)
    for fold in sorted(fold_ids):
        train, _ = folds[fold]
        train_classes, train_counts = np.unique(y_arr[train], return_counts=True)
        if len(train_classes) != len(all_classes):
            missing = np.setdiff1d(all_classes, train_classes)
            raise ValueError(
                f"Fold {fold} training set is missing class(es) {missing.tolist()}; "
                "all classes in y must appear in each selected fold's training set "
                "so label encodings match across folds."
            )
        if require_inner_cv and int(np.min(train_counts)) < 2:
            raise ValueError(
                f"Fold {fold} training set has a class with fewer than 2 cases, "
                "which is insufficient for inner cross-validation when "
                "build_train_file=True."
            )


def _validate_regression_folds(folds, fold_ids, require_inner_cv):
    """Ensure selected regression folds have enough cases for inner CV when needed."""
    if not require_inner_cv:
        return
    for fold in sorted(fold_ids):
        train, _ = folds[fold]
        if len(train) < 2:
            raise ValueError(
                f"Fold {fold} training set must contain at least 2 cases for inner "
                "cross-validation when build_train_file=True."
            )


def _normalise_fold_ids(fold_ids, n_folds):
    """Validate fold ids and return them as a set for membership checks."""
    if n_folds < 1:
        raise ValueError("Cross-validation must contain at least one fold.")
    if fold_ids is None:
        return set(range(n_folds))
    if isinstance(fold_ids, Integral) and not isinstance(fold_ids, bool):
        fold_ids = [int(fold_ids)]
    elif isinstance(fold_ids, (str, bytes, bytearray, Mapping)):
        raise TypeError("fold_ids must be an integer or an iterable of integers.")
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
    if hasattr(indices, "to_numpy"):
        indices = indices.to_numpy()
    if _is_pd_multiindex_collection(data):
        indices = np.asarray(indices)
        if indices.dtype == bool:
            indices = np.flatnonzero(indices)
        if len(indices) == 0:
            return data.iloc[0:0]
        case_ids = data.index.get_level_values(0).unique()
        return pd.concat(
            [data.xs(case_ids[int(index)], level=0) for index in indices],
            keys=range(len(indices)),
            names=data.index.names,
        )
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


def _validate_file_options(
    build_test_file,
    build_train_file,
    data_transforms,
    transform_train_only=False,
):
    """Validate output options before any folds are fitted or files are written."""
    if not build_test_file and not build_train_file:
        raise ValueError(
            "Both test_file and train_file are set to False. At least one must be "
            "written."
        )
    normalised_transforms = _normalise_data_transforms(
        data_transforms, transform_train_only=transform_train_only
    )
    if build_train_file and normalised_transforms is not None:
        raise ValueError(
            "build_train_file=True cannot be combined with data_transforms. "
            "External transforms are fitted before the inner train-estimation "
            "splits and can leak validation information. Set build_train_file=False "
            "or use an estimator pipeline that refits preprocessing within each "
            "train-estimation split."
        )
    return normalised_transforms


def _normalise_data_transforms(data_transforms, transform_train_only=False):
    """Validate data_transforms and normalise sequences to a non-empty list or None."""
    if data_transforms is None:
        return None
    if isinstance(data_transforms, (list, tuple)):
        if len(data_transforms) == 0:
            return None
        transforms = list(data_transforms)
        for transform in transforms:
            _validate_single_transform(transform, transform_train_only)
        return transforms
    _validate_single_transform(data_transforms, transform_train_only)
    return data_transforms


def _validate_single_transform(transform, transform_train_only):
    """Validate that an object implements the required transformer methods."""
    if isinstance(transform, type) or not callable(
        getattr(transform, "fit_transform", None)
    ):
        raise TypeError(
            "data_transforms must be a transformer or a list/tuple of transformers "
            "implementing fit_transform."
        )
    if not transform_train_only and not callable(getattr(transform, "transform", None)):
        raise TypeError(
            "Each transformer in data_transforms must implement transform when "
            "transform_train_only=False."
        )
