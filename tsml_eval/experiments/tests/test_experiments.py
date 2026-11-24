"""General tests for the experiments module."""

import os

import numpy as np
import pandas as pd
import pytest
from aeon.classification import DummyClassifier
from aeon.clustering import DummyClusterer
from aeon.regression import DummyRegressor
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection import BaseCollectionTransformer

from tsml_eval.experiments import (
    classification_experiments,
    load_and_run_classification_experiment,
    load_and_run_clustering_experiment,
    load_and_run_regression_experiment,
    run_classification_experiment,
    run_regression_experiment,
)
from tsml_eval.experiments.tests import _CLASSIFIER_RESULTS_PATH
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH, _TEST_OUTPUT_PATH
from tsml_eval.utils.tests.test_results_writing import _check_classification_file_format


def test_kwargs():
    """Test experiments with kwargs input."""
    dataset = "MinimalChinatown"
    classifier = "LogisticRegression"

    result_path = _TEST_OUTPUT_PATH + "/kwargs/"

    args = [
        _TEST_DATA_PATH,
        result_path,
        classifier,
        dataset,
        "0",
        "--kwargs",
        "fit_intercept",
        "False",
        "bool",
        "--kwargs",
        "C",
        "0.8",
        "float",
        "--kwargs",
        "max_iter",
        "10",
        "int",
        "-ow",
    ]

    classification_experiments.run_experiment(args)

    test_file = f"{result_path}{classifier}/Predictions/{dataset}/testResample0.csv"

    assert os.path.exists(test_file)
    os.remove(test_file)


def test_experiments_predefined_resample_data_loading():
    """Test experiments with data loading."""
    dataset = "PredefinedChinatown"

    load_and_run_classification_experiment(
        _TEST_DATA_PATH + "_test_data/",
        _CLASSIFIER_RESULTS_PATH,
        dataset,
        DummyClassifier(),
        resample_id=5,
        predefined_resample=True,
    )

    test_file = (
        f"{_CLASSIFIER_RESULTS_PATH}/DummyClassifier/Predictions/{dataset}/"
        "testResample5.csv"
    )
    assert os.path.exists(test_file)
    _check_classification_file_format(test_file)

    os.remove(test_file)


@pytest.mark.parametrize(
    "load_and_run, estimator, dataset",
    [
        (load_and_run_classification_experiment, DummyClassifier(), "MinimalChinatown"),
        (load_and_run_regression_experiment, DummyRegressor(), "MinimalGasPrices"),
        (load_and_run_clustering_experiment, DummyClusterer(), "MinimalChinatown"),
    ],
)
def test_attributes_written_per_resample(load_and_run, estimator, dataset, tmp_path):
    """Test each resample writes its attributes to its own directory."""
    for resample_id in [0, 1]:
        load_and_run(
            _TEST_DATA_PATH,
            str(tmp_path),
            dataset,
            estimator,
            resample_id=resample_id,
            write_attributes=True,
            benchmark_time=False,
        )

    name = type(estimator).__name__
    workspace = tmp_path / name / "Workspace" / dataset
    assert sorted(os.listdir(workspace)) == ["Resample0", "Resample1"]
    assert (workspace / "Resample0" / f"{name}.txt").exists()
    assert (workspace / "Resample1" / f"{name}.txt").exists()


class _RecordingTransformer(BaseCollectionTransformer):
    """Records the case ids (first value of each series) seen in fit and transform."""

    calls: list[tuple[set, set]] = []

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        self.fit_ids_ = set(X[:, 0, 0])
        return self

    def _transform(self, X, y=None):
        _RecordingTransformer.calls.append((self.fit_ids_, set(X[:, 0, 0])))
        return X


@pytest.mark.parametrize(
    "run_experiment, estimator, kwargs",
    [
        (run_classification_experiment, DummyClassifier(), {}),
        (
            run_classification_experiment,
            DummyClassifier(),
            {"transform_train_only": True},
        ),
        (run_regression_experiment, DummyRegressor(), {}),
    ],
)
def test_train_estimate_refits_data_transforms(
    run_experiment, estimator, kwargs, tmp_path
):
    """Test the CV train estimate fits data transforms on inner train cases only."""
    X, y = make_example_3d_numpy(
        n_cases=20,
        regression_target=run_experiment is run_regression_experiment,
        random_state=0,
    )
    X[:, 0, 0] = np.arange(20)
    X_test = X.copy()
    X_test[:, 0, 0] += 100
    _RecordingTransformer.calls = []

    run_experiment(
        X,
        y,
        X_test,
        y,
        estimator,
        str(tmp_path),
        data_transforms=_RecordingTransformer(),
        build_test_file=False,
        build_train_file=True,
        benchmark_time=False,
        **kwargs,
    )

    train_ids = set(range(20))
    calls = _RecordingTransformer.calls
    fold_fits = [fit for fit, ids in calls if fit == ids and fit < train_ids]
    held_out = [(fit, ids) for fit, ids in calls if fit != ids and ids <= train_ids]

    # every train case is left out of exactly one fold fit
    assert sorted(i for fit in fold_fits for i in train_ids - fit) == sorted(train_ids)
    # each validation part is transformed by the fit that left it out
    if kwargs.get("transform_train_only"):
        assert held_out == []
    else:
        assert held_out == [(fit, train_ids - fit) for fit in fold_fits]


class _StatelessRecordingTransformer(BaseCollectionTransformer):
    """Records the number of cases in each transform call, with no fitted state."""

    _tags = {"fit_is_empty": True}
    calls: list[int] = []

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        _StatelessRecordingTransformer.calls.append(len(X))
        return X


@pytest.mark.parametrize(
    "run_experiment, estimator",
    [
        (run_classification_experiment, DummyClassifier()),
        (run_regression_experiment, DummyRegressor()),
    ],
)
def test_train_estimate_does_not_refit_fit_is_empty_transforms(
    run_experiment, estimator, tmp_path
):
    """Test transforms with no fitted state are applied once, not in each fold."""
    X, y = make_example_3d_numpy(
        n_cases=20,
        regression_target=run_experiment is run_regression_experiment,
        random_state=0,
    )
    _StatelessRecordingTransformer.calls = []

    run_experiment(
        X,
        y,
        X,
        y,
        estimator,
        str(tmp_path),
        data_transforms=_StatelessRecordingTransformer(),
        build_test_file=False,
        build_train_file=True,
        benchmark_time=False,
    )

    assert _StatelessRecordingTransformer.calls == [20, 20]


@pytest.mark.parametrize(
    "run_experiment, estimator",
    [
        (run_classification_experiment, DummyClassifier()),
        (run_regression_experiment, DummyRegressor()),
    ],
)
def test_train_estimate_with_dataframe_input(run_experiment, estimator, tmp_path):
    """Test a pandas collection is converted before the CV train estimate."""
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 12)))
    y = (
        np.tile([0, 1], 10)
        if run_experiment is run_classification_experiment
        else (np.arange(20, dtype=float))
    )

    run_experiment(
        X,
        y,
        X,
        y,
        estimator,
        str(tmp_path),
        build_test_file=False,
        build_train_file=True,
        benchmark_time=False,
    )

    assert os.path.exists(
        tmp_path / type(estimator).__name__ / "Predictions" / "trainResults.csv"
    )
