"""Tests for cross-validation functions."""

import os

import numpy as np
import pandas as pd
import pytest
from aeon.classification import DummyClassifier
from aeon.datasets import load_covid_3month, load_unit_test
from aeon.regression import DummyRegressor
from aeon.transformations.collection import Normalizer
from aeon.utils.conversion import convert_collection
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier as SklearnDummyClassifier
from sklearn.dummy import DummyRegressor as SklearnDummyRegressor
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from tsml_eval.experiments import cross_validation as cv_module
from tsml_eval.experiments.cross_validation import (
    classification_cross_validation,
    classification_cross_validation_folds,
    regression_cross_validation,
    regression_cross_validation_folds,
)
from tsml_eval.experiments.tests import (
    _CLASSIFIER_RESULTS_PATH,
    _REGRESSOR_RESULTS_PATH,
)
from tsml_eval.utils.tests.test_results_writing import (
    _check_classification_file_format,
    _check_regression_file_format,
)


def test_classification_cross_validation():
    """Test the classification cross-validation function."""
    X, y = load_unit_test()
    classification_cross_validation(
        X,
        y,
        DummyClassifier(),
        _CLASSIFIER_RESULTS_PATH,
        classifier_name="DummyClassifierCV",
        benchmark_time=False,
        overwrite=True,
    )

    for i in range(10):
        test_file = (
            f"{_CLASSIFIER_RESULTS_PATH}/DummyClassifierCV/Predictions/"
            f"testResample{i}.csv"
        )

        assert os.path.exists(test_file)
        _check_classification_file_format(test_file)

    folds = classification_cross_validation_folds(X, y)
    assert len(folds) == 10
    assert len(folds[0]) == 2
    assert isinstance(folds[0][0], np.ndarray)
    assert isinstance(folds[0][1], np.ndarray)


def test_regression_cross_validation():
    """Test the regression cross-validation function."""
    X, y = load_covid_3month()
    regression_cross_validation(
        X,
        y,
        DummyRegressor(),
        _REGRESSOR_RESULTS_PATH,
        regressor_name="DummyRegressorCV",
        benchmark_time=False,
        data_transforms=Normalizer(),
        overwrite=True,
    )

    for i in range(10):
        test_file = (
            f"{_REGRESSOR_RESULTS_PATH}/DummyRegressorCV/Predictions/"
            f"testResample{i}.csv"
        )

        assert os.path.exists(test_file)
        _check_regression_file_format(test_file)

    folds = regression_cross_validation_folds(X, y)
    assert len(folds) == 10
    assert len(folds[0]) == 2
    assert isinstance(folds[0][0], np.ndarray)
    assert isinstance(folds[0][1], np.ndarray)


def test_classification_cross_validation_reduces_default_folds(tmp_path):
    """Test that both classification CV functions reduce folds for small classes."""
    X = np.zeros((12, 1, 3))
    y = np.array([0] * 6 + [1] * 6)

    folds = classification_cross_validation_folds(X, y)
    assert len(folds) == 6

    classification_cross_validation(
        X,
        y,
        DummyClassifier(),
        str(tmp_path),
        classifier_name="SmallClassCV",
        dataset_name="Small",
        benchmark_time=False,
    )
    preds_dir = tmp_path / "SmallClassCV" / "Predictions" / "Small"
    for fold in range(6):
        _check_classification_file_format(
            str(preds_dir / f"testResample{fold}.csv"), num_results_lines=2
        )
    assert not (preds_dir / "testResample6.csv").exists()


class _RecordingTransform(BaseEstimator):
    """Learn an offset and record training cases, rejecting fitted-state reuse."""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        """Fit the offset on training cases and transform X."""
        assert not hasattr(self, "fit_ids_")
        self.fit_ids_ = tuple(int(series[0, 0]) for series in X)
        self.offset_ = np.mean(self.fit_ids_)
        return self.transform(X)

    def transform(self, X, y=None):
        """Subtract the learned training offset from each series."""
        return [series - self.offset_ for series in X]


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize(
    ("random_seed", "expected_seeds"),
    [(None, [0, 1]), (np.int64(5), [5, 5]), ([3, 4], [3, 4])],
)
def test_cross_validation_clones_and_indexes_list_data(
    tmp_path, monkeypatch, task, random_seed, expected_seeds
):
    """Check estimators and transforms are fitted per fold on unequal length data."""
    X = [np.full((1, 3 + i % 3), float(i)) for i in range(12)]
    y = np.tile([0, 1], 6) if task == "classification" else np.arange(12, dtype=float)
    estimator = DummyClassifier() if task == "classification" else DummyRegressor()
    transform = _RecordingTransform()
    run_cv = getattr(cv_module, f"{task}_cross_validation")
    folds = getattr(cv_module, f"{task}_cross_validation_folds")(X, y, cv=2)
    runner_name = f"run_{task}_experiment"
    runner = getattr(cv_module, runner_name)
    fitted = []

    def record_fold(*args, **kwargs):
        fitted.append((args[4], kwargs["data_transforms"]))
        return runner(*args, **kwargs)

    monkeypatch.setattr(cv_module, runner_name, record_fold)
    run_cv(
        X,
        y,
        estimator,
        str(tmp_path / "results"),
        cv=2,
        dataset_name="Unequal",
        attribute_file_path=str(tmp_path / "attributes"),
        benchmark_time=False,
        data_transforms=transform,
        random_seed=random_seed,
    )

    assert len(fitted) == 2
    assert not estimator.is_fitted
    assert not hasattr(transform, "fit_ids_")
    assert transform.random_state is None

    name = type(estimator).__name__
    check_format = (
        _check_classification_file_format
        if task == "classification"
        else _check_regression_file_format
    )
    for fold, (train, test) in enumerate(folds):
        fold_estimator, fold_transform = fitted[fold]
        assert fold_estimator is not estimator
        assert fold_estimator.is_fitted
        seed = expected_seeds[fold]
        assert fold_estimator.get_params().get("random_state", seed) == seed
        assert fold_transform.random_state == seed
        assert fold_transform.fit_ids_ == tuple(train)
        test_file = (
            tmp_path
            / "results"
            / name
            / "Predictions"
            / "Unequal"
            / f"testResample{fold}.csv"
        )
        check_format(str(test_file), num_results_lines=len(test))
        assert (tmp_path / "attributes" / f"fold{fold}" / f"{name}.txt").exists()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cross_validation_pandas_input(tmp_path, task):
    """Test a pandas Series target is indexed by position, not by its index."""
    n_cases = 8
    X = pd.DataFrame(np.arange(n_cases * 4, dtype=float).reshape(n_cases, 4))
    raw_y = (
        [0, 1, 0, 1, 0, 1, 0, 1]
        if task == "classification"
        else [10.1, 20.2, 30.3, 40.4, 50.5, 60.6, 70.7, 80.8]
    )
    y = pd.Series(raw_y, index=[7, 6, 5, 4, 3, 2, 1, 0])

    splits = [([0, 1, 2, 3], [4, 5, 6, 7]), ([4, 5, 6, 7], [0, 1, 2, 3])]
    estimator = (
        SklearnDummyClassifier()
        if task == "classification"
        else SklearnDummyRegressor()
    )
    getattr(cv_module, f"{task}_cross_validation")(
        X,
        y,
        estimator,
        str(tmp_path),
        cv=splits,
        dataset_name="PandasTargets",
        benchmark_time=False,
    )

    preds_dir = tmp_path / type(estimator).__name__ / "Predictions" / "PandasTargets"
    for fold, (_, test_idx) in enumerate(splits):
        test_file = preds_dir / f"testResample{fold}.csv"
        if task == "classification":
            _check_classification_file_format(str(test_file), num_results_lines=4)
        else:
            _check_regression_file_format(str(test_file), num_results_lines=4)
            lines = test_file.read_text(encoding="utf-8").splitlines()[3:]
            written_actuals = [float(line.split(",")[0]) for line in lines]
            expected_actuals = [raw_y[idx] for idx in test_idx]
            assert np.allclose(written_actuals, expected_actuals)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("data_type", ["df-list", "pd-multiindex"])
def test_cross_validation_dataframe_collections(tmp_path, task, data_type):
    """Support unequal length df-list and pd-multiindex collections."""
    X_list = [np.full((2, 3 + i % 3), float(i)) for i in range(8)]
    X = convert_collection(X_list, data_type)
    y = (
        np.array([0, 1, 0, 1, 0, 1, 0, 1])
        if task == "classification"
        else np.arange(8, dtype=float) * 1.5
    )
    estimator = DummyClassifier() if task == "classification" else DummyRegressor()

    getattr(cv_module, f"{task}_cross_validation")(
        X,
        y,
        estimator,
        str(tmp_path),
        cv=2,
        dataset_name="DFCollection",
        benchmark_time=False,
    )
    preds_dir = tmp_path / type(estimator).__name__ / "Predictions" / "DFCollection"
    check_format = (
        _check_classification_file_format
        if task == "classification"
        else _check_regression_file_format
    )
    for fold in range(2):
        check_format(str(preds_dir / f"testResample{fold}.csv"), num_results_lines=4)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cross_validation_train_file(tmp_path, task):
    """Test train files are written for a selected fold."""
    estimator = DummyClassifier() if task == "classification" else DummyRegressor()
    y = np.tile([0, 1], 6) if task == "classification" else np.arange(12) / 10
    getattr(cv_module, f"{task}_cross_validation")(
        np.zeros((12, 1, 3)),
        y,
        estimator,
        str(tmp_path),
        cv=2,
        fold_ids=np.int64(0),
        build_test_file=False,
        build_train_file=True,
        benchmark_time=False,
    )
    directory = tmp_path / type(estimator).__name__ / "Predictions"
    check_format = (
        _check_classification_file_format
        if task == "classification"
        else _check_regression_file_format
    )
    check_format(str(directory / "trainResample0.csv"), num_results_lines=6)
    first_line = (directory / "trainResample0.csv").read_text().splitlines()[0]
    assert "Data transformers: None." in first_line
    assert f"Cross-validation fold 0 of 2 from {task}_cross_validation." in first_line
    assert not (directory / "testResample0.csv").exists()
    assert not (directory / "trainResample1.csv").exists()


def test_classification_cross_validation_missing_class(tmp_path):
    """Test folds with a class missing from the training set are rejected."""
    X = np.zeros((6, 1, 3))
    y = np.array([0, 0, 1, 1, 2, 2])
    splits = [([0, 2, 4], [1, 3, 5]), ([0, 2], [1, 3])]

    with pytest.raises(ValueError, match="missing class"):
        classification_cross_validation(
            X,
            y,
            DummyClassifier(),
            str(tmp_path / "missing"),
            cv=splits,
            benchmark_time=False,
        )
    assert not (tmp_path / "missing").exists()

    classification_cross_validation(
        X,
        y,
        DummyClassifier(),
        str(tmp_path / "fold0"),
        cv=splits,
        fold_ids=0,
        benchmark_time=False,
    )
    assert (
        tmp_path / "fold0" / "DummyClassifier" / "Predictions" / "testResample0.csv"
    ).exists()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cross_validation_groups_and_precomputed_splits(tmp_path, task):
    """Support group-aware splitters and pre-computed splits."""
    X = np.zeros((12, 1, 3))
    y = np.tile([0, 1], 6) if task == "classification" else np.arange(12) / 10
    groups = np.repeat(["g0", "g1", "g2"], 4)
    splitter = (
        StratifiedGroupKFold(n_splits=3)
        if task == "classification"
        else GroupKFold(n_splits=3)
    )
    get_folds = getattr(cv_module, f"{task}_cross_validation_folds")
    run_cv = getattr(cv_module, f"{task}_cross_validation")
    estimator = DummyClassifier() if task == "classification" else DummyRegressor()

    folds = get_folds(X, y, cv=splitter, groups=groups)
    assert len(folds) == 3
    for train, test in folds:
        assert set(groups[train]).isdisjoint(groups[test])

    run_cv(
        X,
        y,
        estimator,
        str(tmp_path / "grouped"),
        cv=splitter,
        groups=groups,
        dataset_name="Grouped",
        benchmark_time=False,
    )
    grouped_dir = (
        tmp_path / "grouped" / type(estimator).__name__ / "Predictions" / "Grouped"
    )
    for fold in range(3):
        assert (grouped_dir / f"testResample{fold}.csv").exists()

    run_cv(
        X,
        y,
        estimator,
        str(tmp_path / "precomputed"),
        cv=folds,
        fold_ids=[0, 2],
        dataset_name="Precomputed",
        benchmark_time=False,
    )
    pre_dir = (
        tmp_path
        / "precomputed"
        / type(estimator).__name__
        / "Predictions"
        / "Precomputed"
    )
    assert (pre_dir / "testResample0.csv").exists()
    assert not (pre_dir / "testResample1.csv").exists()
    assert (pre_dir / "testResample2.csv").exists()

    with pytest.warns(UserWarning, match="ignored by pre-computed splits"):
        get_folds(X, y, cv=folds, groups=groups)


def test_cross_validation_precomputed_split_checks():
    """Test boolean masks become indices and invalid splits are rejected."""
    X = [np.full((1, 3 + i % 3), float(i)) for i in range(8)]
    y = np.tile([0, 1], 4)
    train_mask = np.arange(8) < 4

    folds = classification_cross_validation_folds(X, y, cv=[(train_mask, ~train_mask)])
    assert folds[0][0].tolist() == [0, 1, 2, 3]
    assert folds[0][1].tolist() == [4, 5, 6, 7]

    with pytest.raises(ValueError, match="overlap"):
        classification_cross_validation_folds(X, y, cv=[([0, 1, 2, 3], [3, 4])])
    with pytest.raises(ValueError, match="between 0 and 7"):
        classification_cross_validation_folds(X, y, cv=[([-1, 0], [1, 2])])
    with pytest.raises(ValueError, match="between 0 and 7"):
        classification_cross_validation_folds(X, y, cv=[([0, 1], [2, 8])])
    with pytest.raises(ValueError, match="length 8"):
        classification_cross_validation_folds(
            X, y, cv=[(train_mask[:6], ~train_mask[:6])]
        )
    with pytest.raises(ValueError, match="must not be empty"):
        classification_cross_validation_folds(X, y, cv=[([0, 1, 2, 3], [])])

    for np_fold, int_fold in zip(
        classification_cross_validation_folds(X, y, cv=np.int64(2)),
        classification_cross_validation_folds(X, y, cv=2),
    ):
        assert np.array_equal(np_fold[1], int_fold[1])


@pytest.mark.parametrize("fold_ids", [-1, [10], [0, 10]])
def test_cross_validation_invalid_fold_ids(tmp_path, fold_ids):
    """Test that out of range fold ids fail before running experiments."""
    X, y = load_unit_test()

    with pytest.raises(ValueError, match="fold_ids must be between"):
        classification_cross_validation(
            X,
            y,
            DummyClassifier(),
            str(tmp_path),
            fold_ids=fold_ids,
            benchmark_time=False,
        )
    assert not list(tmp_path.iterdir())


def test_classification_cross_validation_invalid_options(tmp_path):
    """Test small classes and disabled result files raise errors."""
    X = np.zeros((4, 1, 3))

    with pytest.raises(ValueError, match="at least 2 values"):
        classification_cross_validation_folds(X, np.array([0, 0, 0, 1]))

    for bad_kwargs in ({"fold_ids": [1.7]}, {"random_seed": "12"}):
        with pytest.raises(TypeError, match="integer or a list of integers"):
            classification_cross_validation(
                X,
                np.array([0, 1, 0, 1]),
                DummyClassifier(),
                str(tmp_path),
                cv=2,
                benchmark_time=False,
                **bad_kwargs,
            )

    with pytest.raises(ValueError, match="one seed per fold"):
        classification_cross_validation(
            X,
            np.array([0, 1, 0, 1]),
            DummyClassifier(),
            str(tmp_path),
            cv=2,
            random_seed=[0, 1, 2],
            benchmark_time=False,
        )

    with pytest.raises(ValueError, match="At least one must be written"):
        classification_cross_validation(
            X,
            np.array([0, 1, 0, 1]),
            DummyClassifier(),
            str(tmp_path),
            cv=2,
            build_test_file=False,
            build_train_file=False,
            benchmark_time=False,
        )


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("dataset_name", ["SkipData", "N/A"])
def test_cross_validation_existing_result_policy(tmp_path, task, dataset_name):
    """Test that existing folds are skipped unless overwrite is requested."""
    X = np.zeros((10, 1, 4))
    y = np.tile([0, 1], 5) if task == "classification" else np.arange(10) / 10
    estimator = DummyClassifier() if task == "classification" else DummyRegressor()
    est_name = f"{type(estimator).__name__}SkipCV"
    preds_dir = tmp_path / est_name / "Predictions"
    if dataset_name != "N/A":
        preds_dir = preds_dir / dataset_name
    test_file = preds_dir / "testResample0.csv"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("existing result", encoding="utf-8")

    run_cv = getattr(cv_module, f"{task}_cross_validation")
    name_kwarg = (
        {"classifier_name": est_name}
        if task == "classification"
        else {"regressor_name": est_name}
    )

    with pytest.warns(UserWarning, match="not overwriting"):
        run_cv(
            X,
            y,
            estimator,
            str(tmp_path),
            cv=2,
            fold_ids=0,
            dataset_name=dataset_name,
            benchmark_time=False,
            **name_kwarg,
        )
    assert test_file.read_text(encoding="utf-8") == "existing result"

    run_cv(
        X,
        y,
        estimator,
        str(tmp_path),
        cv=2,
        fold_ids=0,
        dataset_name=dataset_name,
        benchmark_time=False,
        overwrite=True,
        **name_kwarg,
    )
    check_format = (
        _check_classification_file_format
        if task == "classification"
        else _check_regression_file_format
    )
    check_format(str(test_file))
