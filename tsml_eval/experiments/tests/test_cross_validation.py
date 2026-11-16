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
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold

from tsml_eval.experiments import cross_validation as cv_module
from tsml_eval.experiments.cross_validation import (
    _normalise_fold_ids,
    _safe_index,
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
    """Test that both classification CV APIs reduce folds for small classes."""
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


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("transform_container", ["single", "list", "tuple"])
def test_cross_validation_clones_and_indexes_list_data(
    tmp_path, monkeypatch, task, transform_container
):
    """Check fitted state and training membership across genuinely unequal folds."""
    X = [np.full((1, 3 + i % 3), float(i)) for i in range(12)]
    y = np.tile([0, 1], 6) if task == "classification" else np.arange(12, dtype=float)
    estimator = DummyClassifier() if task == "classification" else DummyRegressor()
    transform = _RecordingTransform()
    if transform_container == "single":
        transforms = transform
    elif transform_container == "list":
        transforms = [transform]
    else:
        transforms = (transform,)
    run_cv = getattr(cv_module, f"{task}_cross_validation")
    folds = getattr(cv_module, f"{task}_cross_validation_folds")(X, y, cv=2)
    runner_name = f"run_{task}_experiment"
    runner = getattr(cv_module, runner_name)
    fitted = []

    def record_fold(*args, **kwargs):
        fold_transform = kwargs["data_transforms"]
        if transform_container in ("list", "tuple"):
            assert isinstance(fold_transform, list)
            fold_transform = fold_transform[0]
        fitted.append((args[4], fold_transform))
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
        data_transforms=transforms,
    )

    assert len(fitted) == 2
    assert fitted[0][0] is not fitted[1][0]
    assert fitted[0][1] is not fitted[1][1]
    assert not estimator.is_fitted
    assert not hasattr(transform, "fit_ids_")
    if task == "classification":
        assert len(estimator.classes_) == 0

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
        assert fold_transform is not transform
        assert fold_transform.fit_ids_ == tuple(train)
        assert set(fold_transform.fit_ids_).isdisjoint(test)
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


class _RecordingTransform(BaseEstimator):
    """Learn an offset and record training cases, rejecting fitted-state reuse."""

    def fit_transform(self, X, y=None):
        """Fit the offset on training cases and transform X."""
        assert not hasattr(self, "fit_ids_")
        self.fit_ids_ = tuple(int(series[0, 0]) for series in X)
        self.offset_ = np.mean(self.fit_ids_)
        return self.transform(X)

    def transform(self, X, y=None):
        """Subtract the learned training offset from each series."""
        return [series - self.offset_ for series in X]


class _TrainOnlyTransform(BaseEstimator):
    """Transformer implementing only fit_transform for transform_train_only=True."""

    def fit_transform(self, X, y=None):
        """Return X unchanged during training transformation."""
        return X


class _CustomEstimateClassifier(DummyClassifier):
    """Dummy classifier advertising capability:train_estimate for inner-CV tests."""

    _tags = {"capability:train_estimate": True}

    def _fit_predict_proba(self, X, y):
        """Fit the dummy classifier and return training probabilities."""
        self._fit(X, y)
        return self._predict_proba(X)


class _CustomEstimateRegressor(DummyRegressor):
    """Dummy regressor advertising capability:train_estimate for inner-CV tests."""

    _tags = {"capability:train_estimate": True}

    def _fit_predict(self, X, y):
        """Fit the dummy regressor and return training predictions."""
        self._fit(X, y)
        return self._predict(X)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("y_kind", ["series_scrambled", "list", "tuple"])
def test_cross_validation_pandas_and_sequence_targets(tmp_path, task, y_kind):
    """Ensure pandas Series and Python list/tuple targets preserve positional order."""
    n_cases = 8
    X = pd.DataFrame(
        np.arange(n_cases * 4, dtype=float).reshape(n_cases, 4),
        index=[10 + i for i in range(n_cases)],
    )
    raw_y = (
        [0, 1, 0, 1, 0, 1, 0, 1]
        if task == "classification"
        else [10.1, 20.2, 30.3, 40.4, 50.5, 60.6, 70.7, 80.8]
    )
    if y_kind == "series_scrambled":
        # Scrambled integer index containing 0..3 in reversed positions to detect
        # accidental label-based indexing in results writing.
        y = pd.Series(raw_y, index=[7, 6, 5, 4, 3, 2, 1, 0])
    elif y_kind == "list":
        y = list(raw_y)
    else:
        y = tuple(raw_y)

    splits = [([0, 1, 2, 3], [4, 5, 6, 7]), ([4, 5, 6, 7], [0, 1, 2, 3])]
    estimator = (
        SklearnDummyClassifier()
        if task == "classification"
        else SklearnDummyRegressor()
    )
    run_cv = getattr(cv_module, f"{task}_cross_validation")
    run_cv(
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
    """Support unequal-length df-list and pd-multiindex aeon collections."""
    X_list = [np.full((2, 3 + i % 3), float(i)) for i in range(8)]
    X = convert_collection(X_list, data_type)
    y = (
        np.array([0, 1, 0, 1, 0, 1, 0, 1])
        if task == "classification"
        else np.arange(8, dtype=float) * 1.5
    )
    estimator = DummyClassifier() if task == "classification" else DummyRegressor()
    run_cv = getattr(cv_module, f"{task}_cross_validation")

    run_cv(
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


def test_safe_index_boolean_masks_and_multiindex():
    """Verify _safe_index supports boolean masks and pd-multiindex collections."""
    mask_np = np.array([True, False, True, False])
    mask_series = pd.Series(mask_np, index=[100, 200, 300, 400])

    arr = np.arange(8).reshape(4, 2)
    df = pd.DataFrame(arr, index=[10, 20, 30, 40])
    lst = [np.full((1, 2 + i), float(i)) for i in range(4)]
    tup = tuple(lst)
    mi = convert_collection(lst, "pd-multiindex")

    for mask in (mask_np, mask_series):
        assert np.array_equal(_safe_index(arr, mask), arr[[0, 2]])
        assert np.array_equal(_safe_index(df, mask).to_numpy(), arr[[0, 2]])
        assert [x[0, 0] for x in _safe_index(lst, mask)] == [0.0, 2.0]
        assert [x[0, 0] for x in _safe_index(tup, mask)] == [0.0, 2.0]
        mi_sub = convert_collection(_safe_index(mi, mask), "np-list")
        assert [x[0, 0] for x in mi_sub] == [0.0, 2.0]

    assert len(_safe_index(mi, np.array([], dtype=int))) == 0


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("transform_container", ["single", "list", "tuple"])
@pytest.mark.parametrize("ignore_custom", [False, True])
def test_cross_validation_rejects_transformed_train_estimates(
    tmp_path, monkeypatch, task, transform_container, ignore_custom
):
    """Reject leaking train estimates before fitting or writing any result files."""

    def unexpected_run(*args, **kwargs):
        pytest.fail("Unsafe transform/train-estimate combination reached the runner")

    monkeypatch.setattr(cv_module, f"run_{task}_experiment", unexpected_run)
    transform = _RecordingTransform()
    if transform_container == "single":
        transforms = transform
    elif transform_container == "list":
        transforms = [transform]
    else:
        transforms = (transform,)
    extra = {"transform_train_only": True} if task == "classification" else {}
    with pytest.raises(ValueError, match="cannot be combined with data_transforms"):
        getattr(cv_module, f"{task}_cross_validation")(
            np.zeros((12, 1, 3)),
            np.tile([0, 1], 6),
            DummyClassifier() if task == "classification" else DummyRegressor(),
            str(tmp_path),
            cv=2,
            build_train_file=True,
            data_transforms=transforms,
            ignore_custom_train_estimate=ignore_custom,
            benchmark_time=False,
            **extra,
        )
    assert not list(tmp_path.iterdir())
    assert not hasattr(transform, "fit_ids_")


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("transforms", [None, [], ()])
def test_cross_validation_train_estimates_without_transforms(
    tmp_path, task, transforms
):
    """Keep train-only estimates available with no external preprocessing."""
    estimator = DummyClassifier() if task == "classification" else DummyRegressor()
    y = np.tile([0, 1], 6) if task == "classification" else np.arange(12) / 10
    getattr(cv_module, f"{task}_cross_validation")(
        np.zeros((12, 1, 3)),
        y,
        estimator,
        str(tmp_path),
        cv=2,
        fold_ids=0,
        build_test_file=False,
        build_train_file=True,
        data_transforms=transforms,
        benchmark_time=False,
    )
    directory = tmp_path / type(estimator).__name__ / "Predictions"
    train_file = directory / "trainResample0.csv"
    check_format = (
        _check_classification_file_format
        if task == "classification"
        else _check_regression_file_format
    )
    check_format(str(train_file), num_results_lines=6)
    if task == "classification":
        first_line = train_file.read_text(encoding="utf-8").splitlines()[0]
        assert "Data transformers: None." in first_line
    assert not (directory / "testResample0.csv").exists()


def test_cross_validation_fold_class_coverage_and_inner_cv_checks(tmp_path):
    """Reject folds missing classes or lacking cases for inner CV before running."""
    X = np.zeros((6, 1, 3))
    # 3 classes; fold 1 omits class 2 from train AND test (e.g. buffered splitter)
    y_cls = np.array([0, 0, 1, 1, 2, 2])
    missing_class_splits = [([0, 2, 4], [1, 3, 5]), ([0, 2], [1, 3])]
    with pytest.raises(ValueError, match="missing class"):
        classification_cross_validation(
            X,
            y_cls,
            DummyClassifier(),
            str(tmp_path / "missing_cls"),
            cv=missing_class_splits,
            benchmark_time=False,
        )
    assert not (tmp_path / "missing_cls").exists()

    # Fold 0 alone IS valid when fold_ids=0 is explicitly selected
    classification_cross_validation(
        X,
        y_cls,
        DummyClassifier(),
        str(tmp_path / "fold0_only"),
        cv=missing_class_splits,
        fold_ids=0,
        benchmark_time=False,
    )
    assert (
        tmp_path
        / "fold0_only"
        / "DummyClassifier"
        / "Predictions"
        / "testResample0.csv"
    ).exists()

    # 1 case per class in outer train: inner CV fails fast, custom estimate succeeds
    X4 = np.zeros((4, 1, 3))
    y4 = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="fewer than 2 cases"):
        classification_cross_validation(
            X4,
            y4,
            DummyClassifier(),
            str(tmp_path / "cls_inner_fail"),
            cv=2,
            build_train_file=True,
            benchmark_time=False,
        )
    assert not (tmp_path / "cls_inner_fail").exists()

    classification_cross_validation(
        X4,
        y4,
        _CustomEstimateClassifier(),
        str(tmp_path / "cls_custom_ok"),
        cv=2,
        fold_ids=0,
        build_train_file=True,
        benchmark_time=False,
    )
    assert (
        tmp_path
        / "cls_custom_ok"
        / "_CustomEstimateClassifier"
        / "Predictions"
        / "trainResample0.csv"
    ).exists()

    # Regression with 1 training case: inner CV fails fast, custom estimate succeeds
    X3 = np.zeros((3, 1, 3))
    y3 = np.array([0.1, 0.2, 0.3])
    tiny_reg_splits = [([0, 1], [2]), ([0], [1, 2])]
    with pytest.raises(ValueError, match="at least 2 cases"):
        regression_cross_validation(
            X3,
            y3,
            DummyRegressor(),
            str(tmp_path / "reg_inner_fail"),
            cv=tiny_reg_splits,
            build_train_file=True,
            benchmark_time=False,
        )
    assert not (tmp_path / "reg_inner_fail").exists()

    regression_cross_validation(
        X3,
        y3,
        _CustomEstimateRegressor(),
        str(tmp_path / "reg_custom_ok"),
        cv=[([0, 1], [2])],
        build_train_file=True,
        benchmark_time=False,
    )
    assert (
        tmp_path
        / "reg_custom_ok"
        / "_CustomEstimateRegressor"
        / "Predictions"
        / "trainResample0.csv"
    ).exists()


def test_cross_validation_transform_and_estimator_validation(tmp_path):
    """Validate estimator and data_transforms types up front."""
    X = np.zeros((8, 1, 3))
    y = np.tile([0, 1], 4)

    with pytest.raises(TypeError, match="classifier must be"):
        classification_cross_validation(
            X, y, DummyRegressor(), str(tmp_path), cv=2, benchmark_time=False
        )

    with pytest.raises(TypeError, match="regressor must be"):
        regression_cross_validation(
            X,
            y.astype(float),
            DummyClassifier(),
            str(tmp_path),
            cv=2,
            benchmark_time=False,
        )

    classification_cross_validation(
        X,
        y,
        DummyClassifier(),
        str(tmp_path / "train_only_ok"),
        cv=2,
        fold_ids=0,
        data_transforms=_TrainOnlyTransform(),
        transform_train_only=True,
        benchmark_time=False,
    )
    assert (
        tmp_path
        / "train_only_ok"
        / "DummyClassifier"
        / "Predictions"
        / "testResample0.csv"
    ).exists()

    with pytest.raises(TypeError, match="must implement transform"):
        classification_cross_validation(
            X,
            y,
            DummyClassifier(),
            str(tmp_path / "missing_transform"),
            cv=2,
            data_transforms=_TrainOnlyTransform(),
            transform_train_only=False,
            benchmark_time=False,
        )

    for bad_transforms in ("invalid", 123, {Normalizer()}, [None], Normalizer):
        with pytest.raises(TypeError, match="data_transforms"):
            regression_cross_validation(
                X,
                y.astype(float),
                DummyRegressor(),
                str(tmp_path / "bad"),
                cv=2,
                data_transforms=bad_transforms,
                benchmark_time=False,
            )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cross_validation_groups_and_iterable_splits(tmp_path, task):
    """Support group-aware splitters and pre-computed split iterables."""
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

    # Pass a generator of boolean masks as pre-computed splits
    bool_gen = (
        (
            np.isin(np.arange(12), train),
            np.isin(np.arange(12), test),
        )
        for train, test in folds
    )
    run_cv(
        X,
        y,
        estimator,
        str(tmp_path / "precomputed"),
        cv=bool_gen,
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

    with pytest.raises(ValueError, match="groups cannot be provided"):
        get_folds(X, y, cv=folds, groups=groups)


@pytest.mark.parametrize(
    ("fold_ids", "error_type"),
    [
        ([], ValueError),
        ([-1], ValueError),
        ([10], ValueError),
        ([0, 0], ValueError),
        (["0"], TypeError),
        ("", TypeError),
        ("0", TypeError),
        (b"\x00\x01", TypeError),
        ({0: "a"}, TypeError),
        (True, TypeError),
        ([True], TypeError),
        (1.5, TypeError),
    ],
)
def test_cross_validation_invalid_fold_ids(fold_ids, error_type):
    """Test that invalid fold selections fail before running experiments."""
    X, y = load_unit_test()

    with pytest.raises(error_type):
        classification_cross_validation(
            X,
            y,
            DummyClassifier(),
            _CLASSIFIER_RESULTS_PATH,
            fold_ids=fold_ids,
            benchmark_time=False,
        )


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize(
    ("bad_cv", "error_type"),
    [
        (1, ValueError),
        (0, ValueError),
        (True, TypeError),
        (1.5, TypeError),
        ("invalid", TypeError),
        ({0: 1}, TypeError),
        (KFold, TypeError),
        ([], ValueError),
        ([0, 1], ValueError),
        (["ab"], ValueError),
        ([([], [0, 1])], ValueError),
        ([([0, 1], [])], ValueError),
        ([([0, 1, 2], [2, 3])], ValueError),
        ([([-1, 0], [1, 2])], ValueError),
        ([([0, 1], [2, 99])], ValueError),
        ([([True, False], [False, True])], ValueError),
        ([([[0, 1]], [2, 3])], ValueError),
        ([([0.5, 1.5], [2, 3])], TypeError),
        ([("01", [2, 3])], TypeError),
    ],
)
def test_cross_validation_invalid_cv(task, bad_cv, error_type):
    """Test that invalid cv specifications raise clear errors."""
    X = np.zeros((6, 1, 3))
    y = np.array([0, 1, 0, 1, 0, 1]) if task == "classification" else np.arange(6.0)
    get_folds = getattr(cv_module, f"{task}_cross_validation_folds")

    with pytest.raises(error_type):
        get_folds(X, y, cv=bad_cv)


def test_cross_validation_invalid_inputs_and_file_options(tmp_path):
    """Test length/ndim checks, class count checks, and file flags."""
    X = np.zeros((4, 1, 3))

    with pytest.raises(ValueError, match="one-dimensional"):
        classification_cross_validation_folds(X, np.zeros((4, 1)))

    with pytest.raises(ValueError, match="same number of cases"):
        classification_cross_validation_folds(X, np.array([0, 1]))

    with pytest.raises(ValueError, match="at least 2 values"):
        classification_cross_validation_folds(X, np.array([0, 0, 0, 1]))

    with pytest.raises(ValueError, match="at least 2 values"):
        classification_cross_validation_folds(np.zeros((0, 1, 3)), np.array([]))

    with pytest.raises(ValueError, match="at least 2 unique classes"):
        classification_cross_validation_folds(X, np.zeros(4, dtype=int))

    with pytest.raises(ValueError, match="at least one fold"):
        _normalise_fold_ids(None, 0)

    default_reg_folds = regression_cross_validation_folds(
        np.zeros((20, 1, 3)), np.arange(20.0)
    )
    assert len(default_reg_folds) == 10

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
def test_cross_validation_existing_result_policy(tmp_path, task):
    """Test that existing folds are skipped unless overwrite is requested."""
    X = np.zeros((10, 1, 4))
    y = np.tile([0, 1], 5) if task == "classification" else np.arange(10) / 10
    estimator = DummyClassifier() if task == "classification" else DummyRegressor()
    est_name = f"{type(estimator).__name__}SkipCV"
    results_path = str(tmp_path)
    test_file = tmp_path / est_name / "Predictions" / "SkipData" / "testResample0.csv"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("existing result", encoding="utf-8")

    run_cv = getattr(cv_module, f"{task}_cross_validation")
    name_kwarg = (
        {"classifier_name": est_name}
        if task == "classification"
        else {"regressor_name": est_name}
    )

    run_cv(
        X,
        y,
        estimator,
        results_path,
        cv=2,
        fold_ids=0,
        dataset_name="SkipData",
        benchmark_time=False,
        **name_kwarg,
    )
    assert test_file.read_text(encoding="utf-8") == "existing result"

    run_cv(
        X,
        y,
        estimator,
        results_path,
        cv=2,
        fold_ids=0,
        dataset_name="SkipData",
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
