"""Functions for running experiments using cross-validation."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "classification_cross_validation",
    "classification_cross_validation_folds",
    "regression_cross_validation",
    "regression_cross_validation_folds",
]

import warnings
from numbers import Integral

import numpy as np
from aeon.utils.conversion import convert_collection
from aeon.utils.validation.collection import get_n_cases, get_type, is_equal_length
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, check_cv

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
    random_seed=None,
):
    """Run a classification experiment using cross-validation.

    Each fold is written with the same file names as a resample from
    ``load_and_run_classification_experiment``, i.e. ``testResample{fold}.csv``, and
    the fold is noted in the comment on the first line of each file. Use a different
    ``results_path`` or ``classifier_name`` to resample experiments to avoid mixing
    the two.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target labels.
    estimator : object
        The classifier to be evaluated. A clone is fitted for each fold, with its
        ``random_state`` set to the fold id (or ``random_seed``) if it has one.
    results_path : str
        Location of where to write results. Any required directories will be created.
    cv : int, splitter object, iterable of splits or None, default=None
        Cross-validation strategy, number of folds, or an iterable of ``(train, test)``
        index splits. If None, up to 10-fold stratified cross-validation will be used.
    fold_ids : int, list of int or None, default=None
        Fold ids to run. If None, all folds will be run.
    classifier_name : str or None, default=None
        Name of classifier used in writing results. If None, the name is taken from
        the classifier.
    dataset_name : str, default="N/A"
        Name of dataset.
    build_test_file : bool, default=True
        Whether to generate test files or not.
    build_train_file : bool, default=False
        Whether to generate train files or not. The train estimate is the classifier's
        own if it can produce one, otherwise a cross-validation on each fold's
        training data with up to 10 folds.
    ignore_custom_train_estimate : bool, default=False
        Whether to ignore the classifier's own train estimate. If False and the
        classifier has the ``capability:train_estimate`` tag, each fold's train file
        uses its own estimate. Otherwise, a cross-validation on that fold's training
        data is used.
    attribute_file_path : str or None, default=None
        Directory to write the fitted classifier's attributes to using
        ``estimator_attributes_to_file``. Each fold writes to a ``fold{i}``
        subfolder, where ``i`` is the fold id. Attributes are only written when the
        fold's test file is built. If None, no attributes are written.
    att_max_shape : int, default=0
        Maximum list shape when writing attributes, passed to
        ``estimator_attributes_to_file`` as ``max_list_shape``. At 0, estimators in
        lists are not written. At 1, estimators in one-dimensional lists are written.
    benchmark_time : bool, default=True
        Whether to benchmark the hardware for each selected fold using a simple
        function and write the result. This will typically take ~2 seconds per fold,
        but is hardware dependent.
    data_transforms : transformer, list of transformers or None, default=None
        Transformer(s) to apply within each fold. Transformers are cloned for each
        fold, with their ``random_state`` set to the fold id (or ``random_seed``) if
        they have one.
    transform_train_only : bool, default=False
        If True, the data transforms are only applied to the training data in each
        fold.
    overwrite : bool, default=False
        If False, existing fold result files are skipped. If True, existing files are
        overwritten.
    groups : array-like or None, default=None
        Group labels for the cases used while splitting the dataset into train and
        test folds. Only used in conjunction with a ``cv`` splitter.
    random_seed : int, list of int or None, default=None
        Use a different random seed than the fold id. An int is used for every fold,
        a list gives one seed per fold. If None, the fold id is used.
    """
    if not build_test_file and not build_train_file:
        raise ValueError(
            "Both test_file and train_file are set to False. At least one must be "
            "written."
        )

    X, y = _convert_X_y(X, y)
    folds = _classification_folds(X, y, cv, groups)
    fold_ids = _check_fold_ids(fold_ids, len(folds))
    seeds = _fold_seeds(random_seed, len(folds))

    classes = np.unique(y)
    for fold in fold_ids:
        missing = np.setdiff1d(classes, y[folds[fold][0]])
        if len(missing) > 0:
            raise ValueError(
                f"Fold {fold} training set is missing class(es) {missing.tolist()}. "
                "All classes must appear in the training set of each selected fold."
            )

    if classifier_name is None:
        classifier_name = type(estimator).__name__

    for fold in fold_ids:
        fold_build_test, fold_build_train = _check_existing_results(
            results_path,
            classifier_name,
            dataset_name,
            fold,
            overwrite,
            build_test_file,
            build_train_file,
        )
        if not fold_build_test and not fold_build_train:
            warnings.warn(
                f"All files exist for fold {fold} and not overwriting, skipping.",
                stacklevel=2,
            )
            continue

        train, test = folds[fold]
        seed = seeds[fold]
        run_classification_experiment(
            _index_cases(X, train),
            y[train],
            _index_cases(X, test),
            y[test],
            _clone_with_seed(estimator, seed),
            results_path,
            classifier_name=classifier_name,
            dataset_name=dataset_name,
            resample_id=fold,
            data_transforms=_clone_transforms(data_transforms, seed),
            transform_train_only=transform_train_only,
            build_test_file=fold_build_test,
            build_train_file=fold_build_train,
            ignore_custom_train_estimate=ignore_custom_train_estimate,
            attribute_file_path=(
                None
                if attribute_file_path is None
                else f"{attribute_file_path}/fold{fold}/"
            ),
            att_max_shape=att_max_shape,
            benchmark_time=benchmark_time,
            extra_comment=(
                f"Cross-validation fold {fold} of {len(folds)} from "
                "classification_cross_validation."
            ),
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
    X, y = _convert_X_y(X, y)
    return _classification_folds(X, y, cv, groups)


def regression_cross_validation(
    X,
    y,
    estimator,
    results_path,
    cv=None,
    fold_ids=None,
    regressor_name=None,
    dataset_name="N/A",
    build_test_file=True,
    build_train_file=False,
    ignore_custom_train_estimate=False,
    attribute_file_path=None,
    att_max_shape=0,
    benchmark_time=True,
    data_transforms=None,
    overwrite=False,
    groups=None,
    random_seed=None,
):
    """Run a regression experiment using cross-validation.

    Each fold is written with the same file names as a resample from
    ``load_and_run_regression_experiment``, i.e. ``testResample{fold}.csv``, and the
    fold is noted in the comment on the first line of each file. Use a different
    ``results_path`` or ``regressor_name`` to resample experiments to avoid mixing
    the two.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target values. Cast to float before running the experiments.
    estimator : object
        The regressor to be evaluated. A clone is fitted for each fold, with its
        ``random_state`` set to the fold id (or ``random_seed``) if it has one.
    results_path : str
        Location of where to write results. Any required directories will be created.
    cv : int, splitter object, iterable of splits or None, default=None
        Cross-validation strategy, number of folds, or an iterable of ``(train, test)``
        index splits. If None, 10-fold shuffled cross-validation will be used.
    fold_ids : int, list of int or None, default=None
        Fold ids to run. If None, all folds will be run.
    regressor_name : str or None, default=None
        Name of regressor used in writing results. If None, the name is taken from
        the regressor.
    dataset_name : str, default="N/A"
        Name of dataset.
    build_test_file : bool, default=True
        Whether to generate test files or not.
    build_train_file : bool, default=False
        Whether to generate train files or not. The train estimate is the regressor's
        own if it can produce one, otherwise a cross-validation on each fold's
        training data with up to 10 folds.
    ignore_custom_train_estimate : bool, default=False
        Whether to ignore the regressor's own train estimate. If False and the
        regressor has the ``capability:train_estimate`` tag, each fold's train file
        uses its own estimate. Otherwise, a cross-validation on that fold's training
        data is used.
    attribute_file_path : str or None, default=None
        Directory to write the fitted regressor's attributes to using
        ``estimator_attributes_to_file``. Each fold writes to a ``fold{i}``
        subfolder, where ``i`` is the fold id. Attributes are only written when the
        fold's test file is built. If None, no attributes are written.
    att_max_shape : int, default=0
        Maximum list shape when writing attributes, passed to
        ``estimator_attributes_to_file`` as ``max_list_shape``. At 0, estimators in
        lists are not written. At 1, estimators in one-dimensional lists are written.
    benchmark_time : bool, default=True
        Whether to benchmark the hardware for each selected fold using a simple
        function and write the result. This will typically take ~2 seconds per fold,
        but is hardware dependent.
    data_transforms : transformer, list of transformers or None, default=None
        Transformer(s) to apply within each fold. Transformers are cloned for each
        fold, with their ``random_state`` set to the fold id (or ``random_seed``) if
        they have one.
    overwrite : bool, default=False
        If False, existing fold result files are skipped. If True, existing files are
        overwritten.
    groups : array-like or None, default=None
        Group labels for the cases used while splitting the dataset into train and
        test folds. Only used in conjunction with a ``cv`` splitter.
    random_seed : int, list of int or None, default=None
        Use a different random seed than the fold id. An int is used for every fold,
        a list gives one seed per fold. If None, the fold id is used.
    """
    if not build_test_file and not build_train_file:
        raise ValueError(
            "Both test_file and train_file are set to False. At least one must be "
            "written."
        )

    X, y = _convert_X_y(X, y)
    y = y.astype(float)
    folds = _regression_folds(X, y, cv, groups)
    fold_ids = _check_fold_ids(fold_ids, len(folds))
    seeds = _fold_seeds(random_seed, len(folds))

    if regressor_name is None:
        regressor_name = type(estimator).__name__

    for fold in fold_ids:
        fold_build_test, fold_build_train = _check_existing_results(
            results_path,
            regressor_name,
            dataset_name,
            fold,
            overwrite,
            build_test_file,
            build_train_file,
        )
        if not fold_build_test and not fold_build_train:
            warnings.warn(
                f"All files exist for fold {fold} and not overwriting, skipping.",
                stacklevel=2,
            )
            continue

        train, test = folds[fold]
        seed = seeds[fold]
        run_regression_experiment(
            _index_cases(X, train),
            y[train],
            _index_cases(X, test),
            y[test],
            _clone_with_seed(estimator, seed),
            results_path,
            regressor_name=regressor_name,
            dataset_name=dataset_name,
            resample_id=fold,
            data_transforms=_clone_transforms(data_transforms, seed),
            build_test_file=fold_build_test,
            build_train_file=fold_build_train,
            ignore_custom_train_estimate=ignore_custom_train_estimate,
            attribute_file_path=(
                None
                if attribute_file_path is None
                else f"{attribute_file_path}/fold{fold}/"
            ),
            att_max_shape=att_max_shape,
            benchmark_time=benchmark_time,
            extra_comment=(
                f"Cross-validation fold {fold} of {len(folds)} from "
                "regression_cross_validation."
            ),
        )


def regression_cross_validation_folds(X, y, cv=None, groups=None):
    """Get the folds for a regression cross-validation experiment.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target values.
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
    X, y = _convert_X_y(X, y)
    return _regression_folds(X, y, cv, groups)


def _convert_X_y(X, y):
    if get_type(X) not in ("numpy3D", "numpy2D", "np-list"):
        X = convert_collection(X, "numpy3D" if is_equal_length(X) else "np-list")
    return X, np.asarray(y)


def _classification_folds(X, y, cv, groups):
    if cv is None:
        _, counts = np.unique(y, return_counts=True)
        cv = min(10, int(np.min(counts)))
        if cv < 2:
            raise ValueError(
                "All classes must have at least 2 values to run the default "
                "cross-validation."
            )
    if isinstance(cv, Integral):
        cv = StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=0)
    return _split(X, y, cv, groups)


def _regression_folds(X, y, cv, groups):
    if cv is None:
        cv = 10
    if isinstance(cv, Integral):
        cv = KFold(n_splits=int(cv), shuffle=True, random_state=0)
    return _split(X, y, cv, groups)


def _split(X, y, cv, groups):
    if groups is not None and not hasattr(cv, "split"):
        warnings.warn(
            "The groups parameter is ignored by pre-computed splits.", stacklevel=4
        )
    n_cases = get_n_cases(X)
    folds = []
    for fold, split in enumerate(check_cv(cv).split(X, y, groups)):
        indices = []
        for idx in map(np.asarray, split):
            if idx.dtype == bool:
                if len(idx) != n_cases:
                    raise ValueError(
                        f"Fold {fold} boolean masks must have length {n_cases}."
                    )
                idx = np.flatnonzero(idx)
            if len(idx) == 0:
                raise ValueError(f"Fold {fold} train and test sets must not be empty.")
            indices.append(idx)
        train, test = indices
        if np.intersect1d(train, test).size > 0:
            raise ValueError(f"Fold {fold} train and test indices overlap.")
        all_idx = np.concatenate([train, test])
        if np.any((all_idx < 0) | (all_idx >= n_cases)):
            raise ValueError(
                f"Fold {fold} indices must be between 0 and {n_cases - 1}."
            )
        folds.append((train, test))
    return folds


def _check_fold_ids(fold_ids, n_folds):
    if fold_ids is None:
        return list(range(n_folds))
    if isinstance(fold_ids, Integral):
        fold_ids = [fold_ids]
    if not all(isinstance(fold, Integral) for fold in fold_ids):
        raise TypeError("fold_ids must be an integer or a list of integers.")
    if any(fold < 0 or fold >= n_folds for fold in fold_ids):
        raise ValueError(f"fold_ids must be between 0 and {n_folds - 1}.")
    return sorted({int(fold) for fold in fold_ids})


def _fold_seeds(random_seed, n_folds):
    if random_seed is None:
        return list(range(n_folds))
    if isinstance(random_seed, Integral):
        return [int(random_seed)] * n_folds
    if isinstance(random_seed, str) or not all(
        isinstance(seed, Integral) for seed in random_seed
    ):
        raise TypeError("random_seed must be an integer or a list of integers.")
    if len(random_seed) != n_folds:
        raise ValueError(
            f"random_seed must have one seed per fold, got {len(random_seed)} seeds "
            f"for {n_folds} folds."
        )
    return [int(seed) for seed in random_seed]


def _index_cases(X, indices):
    if isinstance(X, list):
        return [X[i] for i in indices]
    return X[indices]


def _clone_with_seed(estimator, seed):
    estimator = clone(estimator)
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    return estimator


def _clone_transforms(data_transforms, seed):
    if data_transforms is None:
        return None
    if isinstance(data_transforms, (list, tuple)):
        return [_clone_with_seed(transform, seed) for transform in data_transforms]
    return _clone_with_seed(data_transforms, seed)
