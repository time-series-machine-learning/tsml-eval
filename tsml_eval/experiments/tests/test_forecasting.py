"""Tests for forecasting experiments."""

import os
import runpy

import pytest
from aeon.classification import DummyClassifier
from aeon.forecasting import NaiveForecaster
from aeon.utils.discovery import all_estimators

from tsml_eval.datasets._test_data._data_sizes import DATA_TEST_SIZES
from tsml_eval.experiments import (
    _get_forecaster,
    forecasting_experiments,
    threaded_forecasting_experiments,
)
from tsml_eval.experiments._get_forecaster import get_forecaster_by_name
from tsml_eval.experiments.experiments import (
    load_and_run_forecasting_experiment,
    run_forecasting_experiment,
)
from tsml_eval.experiments.tests import _FORECASTER_RESULTS_PATH
from tsml_eval.testing.testing_utils import (
    _TEST_DATA_PATH,
    _check_set_method,
    _check_set_method_results,
)
from tsml_eval.utils.tests.test_results_writing import _check_forecasting_file_format


@pytest.mark.parametrize(
    "module", [forecasting_experiments, threaded_forecasting_experiments]
)
def test_forecasting_experiment_overwrite(module, monkeypatch, tmp_path):
    """Test the forecasting scripts only rerun present results with --overwrite."""
    predictions = tmp_path / "DummyForecaster" / "Predictions" / "ShampooSales"
    predictions.mkdir(parents=True)
    (predictions / "testResample0.csv").touch()

    calls = []
    monkeypatch.setattr(module, "get_forecaster_by_name", lambda *a, **kw: None)
    monkeypatch.setattr(
        module,
        "load_and_run_forecasting_experiment",
        lambda *a, **kw: calls.append(kw),
    )

    args = [_TEST_DATA_PATH, str(tmp_path), "DummyForecaster", "ShampooSales", "0"]
    module.run_experiment(args)
    assert calls == []

    module.run_experiment(args + ["-ow"])
    assert len(calls) == 1


def test_run_forecasting_experiment():
    """Test forecasting experiments with test data and forecasting."""
    forecaster = "NaiveForecaster"
    dataset = "ShampooSales"

    args = [
        _TEST_DATA_PATH,
        _FORECASTER_RESULTS_PATH,
        forecaster,
        dataset,
        "2",
    ]

    forecasting_experiments.run_experiment(args)

    test_file = (
        f"{_FORECASTER_RESULTS_PATH}{forecaster}/Predictions/{dataset}/"
        "testResample2.csv"
    )
    assert os.path.exists(test_file)
    _check_forecasting_file_format(
        test_file, num_results_lines=DATA_TEST_SIZES[dataset]
    )

    # test present results checking
    forecasting_experiments.run_experiment(args)

    os.remove(test_file)


def test_run_forecasting_experiment_main():
    """Test forecasting experiments main with test data and forecaster."""
    forecaster = "NaiveForecaster"
    dataset = "ShampooSales"

    # run twice to test results present check
    for _ in range(2):
        runpy.run_path(
            (
                "./tsml_eval/experiments/forecasting_experiments.py"
                if os.getcwd().split("\\")[-1] != "tests"
                else "../forecasting_experiments.py"
            ),
            run_name="__main__",
        )

    test_file = (
        f"{_FORECASTER_RESULTS_PATH}{forecaster}/Predictions/{dataset}/"
        "testResample0.csv"
    )
    assert os.path.exists(test_file)
    _check_forecasting_file_format(test_file)

    os.remove(test_file)


def test_run_threaded_forecasting_experiment():
    """Test threaded forecasting experiments with test data and forecaster."""
    forecaster = "NaiveForecaster"
    dataset = "ShampooSales"

    args = [
        _TEST_DATA_PATH,
        _FORECASTER_RESULTS_PATH,
        forecaster,
        dataset,
        "1",
        "-nj",
        "2",
        # also test benchmark time here
        "--benchmark_time",
    ]

    threaded_forecasting_experiments.run_experiment(args)

    test_file = (
        f"{_FORECASTER_RESULTS_PATH}{forecaster}/Predictions/{dataset}/"
        "testResample1.csv"
    )
    assert os.path.exists(test_file)
    _check_forecasting_file_format(test_file)

    # test present results checking
    threaded_forecasting_experiments.run_experiment(args)

    # this covers the main method and experiment function result file checking
    runpy.run_path(
        (
            "./tsml_eval/experiments/threaded_forecasting_experiments.py"
            if os.getcwd().split("\\")[-1] != "tests"
            else "../threaded_forecasting_experiments.py"
        ),
        run_name="__main__",
    )

    os.remove(test_file)


def test_threaded_forecasting_experiment_arguments(monkeypatch, tmp_path):
    """Test the threaded script passes the same arguments as the unthreaded one."""
    args = [
        _TEST_DATA_PATH,
        str(tmp_path),
        "NaiveForecaster",
        "ShampooSales",
        "0",
        "-fs",
        "direct",
    ]

    calls = []
    for module in (forecasting_experiments, threaded_forecasting_experiments):
        monkeypatch.setattr(
            module, "get_forecaster_by_name", lambda *a, **kw: calls.append(kw)
        )
        monkeypatch.setattr(
            module,
            "load_and_run_forecasting_experiment",
            lambda *a, **kw: calls.append(kw),
        )
        module.run_experiment(args)

    getter, run, threaded_getter, threaded_run = calls
    assert threaded_getter == getter
    assert threaded_run == run


@pytest.mark.parametrize(
    "forecasting_strategy", ["iterative", "direct", "series_to_series"]
)
def test_run_forecasting_experiment_strategies(forecasting_strategy, tmp_path):
    """Test forecasting experiments with each forecasting strategy."""
    if forecasting_strategy == "series_to_series":
        pytest.importorskip("tensorflow")
        from aeon.forecasting.deep_learning import NBeatsForecaster

        forecaster = NBeatsForecaster(
            window=10, n_epochs=2, random_state=0, file_path=f"{tmp_path}/"
        )
    else:
        forecaster = NaiveForecaster()

    load_and_run_forecasting_experiment(
        _TEST_DATA_PATH,
        str(tmp_path),
        "ShampooSales",
        forecaster,
        random_seed=0,
        benchmark_time=False,
        forecasting_strategy=forecasting_strategy,
    )

    test_file = (
        f"{tmp_path}/{type(forecaster).__name__}/Predictions/ShampooSales/"
        "testResample0.csv"
    )
    assert os.path.exists(test_file)
    _check_forecasting_file_format(
        test_file, num_results_lines=DATA_TEST_SIZES["ShampooSales"]
    )


def test_run_forecasting_experiment_invalid_forecasting_strategy():
    """Test run_forecasting_experiment method with invalid forecasting strategies."""
    with pytest.raises(ValueError, match="forecasting_strategy must be one of"):
        run_forecasting_experiment(
            [],
            [],
            NaiveForecaster(),
            [],
            forecasting_strategy="invalid",
        )

    with pytest.raises(ValueError, match="does not support the 'series_to_series'"):
        run_forecasting_experiment(
            [],
            [],
            NaiveForecaster(),
            [],
            forecasting_strategy="series_to_series",
        )


def test_run_forecasting_experiment_invalid_estimator():
    """Test run_forecasting_experiment method with invalid estimator."""
    with pytest.raises(TypeError, match="forecaster must be a"):
        run_forecasting_experiment(
            [],
            [],
            DummyClassifier(),
            [],
        )


def _get_forecaster_by_name_with_window(forecaster_name):
    if forecaster_name.lower() in ["tcnforecaster", "tcn", "tvpforecaster", "tvp"]:
        return get_forecaster_by_name(forecaster_name, window=10)
    return get_forecaster_by_name(forecaster_name)


def test_get_forecaster_by_name():
    """Test get_forecaster_by_name method."""
    forecaster_name_lists = [
        _get_forecaster.deep_forecasters,
        _get_forecaster.ml_forecasters,
        _get_forecaster.other_forecasters,
        _get_forecaster.regression_forecasters,
        _get_forecaster.stats_forecasters,
    ]

    forecaster_list = []
    forecaster_dict = {}
    all_forecaster_names = []
    for forecaster_name_list in forecaster_name_lists:
        _check_set_method(
            _get_forecaster_by_name_with_window,
            forecaster_name_list,
            forecaster_list,
            forecaster_dict,
            all_forecaster_names,
        )

    _check_set_method_results(
        forecaster_dict,
        estimator_name="Forecasters",
        method_name="get_forecaster_by_name",
    )


def test_get_forecaster_by_name_invalid():
    """Test get_forecaster_by_name method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN FORECASTER"):
        get_forecaster_by_name("invalid")


def test_aeon_forecasters_available():
    """Test all aeon forecasters are available."""
    excluded = [
        # composable/wrapper
        "EnsembleForecaster",
        "RegressionForecaster",
        # just missing
        "AutoCES",
        "AutoETS",
        "CES",
        "DOTM",
        "DeepARForecaster",
        "NBeatsForecaster",
        "SCUM",
    ]

    est = [e for e, _ in all_estimators(type_filter="forecaster")]
    for e in est:
        if e in excluded:
            continue

        try:
            assert _get_forecaster_by_name_with_window(e) is not None
        except ModuleNotFoundError:
            continue


def test_forecasting_attributes_written_per_seed(tmp_path):
    """Test each random seed writes its attributes to its own directory."""
    for random_seed in [0, 1, None]:
        load_and_run_forecasting_experiment(
            _TEST_DATA_PATH,
            str(tmp_path),
            "ShampooSales",
            NaiveForecaster(),
            random_seed=random_seed,
            write_attributes=True,
            benchmark_time=False,
        )

    workspace = tmp_path / "NaiveForecaster" / "Workspace" / "ShampooSales"
    assert sorted(os.listdir(workspace)) == ["Resample0", "Resample1", "Results"]
    assert (workspace / "Resample0" / "NaiveForecaster.txt").exists()
    assert (workspace / "Resample1" / "NaiveForecaster.txt").exists()
    assert (workspace / "Results" / "NaiveForecaster.txt").exists()
