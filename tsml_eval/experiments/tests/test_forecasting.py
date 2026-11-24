"""Tests for forecasting experiments."""

import pytest
from aeon.forecasting import NaiveForecaster

from tsml_eval.experiments import (
    experiments,
    forecasting_experiments,
    threaded_forecasting_experiments,
)
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH


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


def test_forecasting_attributes_written_per_seed(monkeypatch, tmp_path):
    """Test each random seed gets its own attribute directory."""
    paths = []
    monkeypatch.setattr(
        experiments,
        "run_forecasting_experiment",
        lambda *a, **kw: paths.append(kw["attribute_file_path"]),
    )

    for random_seed in [0, 1, None]:
        experiments.load_and_run_forecasting_experiment(
            _TEST_DATA_PATH,
            str(tmp_path),
            "ShampooSales",
            NaiveForecaster(),
            random_seed=random_seed,
            write_attributes=True,
        )

    workspace = f"{tmp_path}/NaiveForecaster/Workspace/ShampooSales"
    assert paths == [
        f"{workspace}/Resample0",
        f"{workspace}/Resample1",
        f"{workspace}/Results",
    ]


# import os
# import runpy
#
# import pytest
# from aeon.utils.discovery import all_estimators
# from tsml.dummy import DummyClassifier
#
# from tsml_eval.datasets._test_data._data_sizes import DATA_TEST_SIZES
# from tsml_eval.experiments import (
#     forecasting_experiments,
#     get_forecaster_by_name,
#     run_forecasting_experiment,
#     set_forecaster,
#     threaded_forecasting_experiments,
# )
# from tsml_eval.experiments.tests import _FORECASTER_RESULTS_PATH
# from tsml_eval.testing.testing_utils import (
#     _TEST_DATA_PATH,
#     _check_set_method,
#     _check_set_method_results,
# )
# from tsml_eval.utils.tests.test_results_writing import _check_forecasting_file_format
#
#
# def test_run_forecasting_experiment():
#     """Test forecasting experiments with test data and forecasting."""
#     forecaster = "DummyForecaster"
#     dataset = "ShampooSales"
#
#     args = [
#         _TEST_DATA_PATH,
#         _FORECASTER_RESULTS_PATH,
#         forecaster,
#         dataset,
#         "2",
#     ]
#
#     forecasting_experiments.run_experiment(args)
#
#     test_file = (
#         f"{_FORECASTER_RESULTS_PATH}{forecaster}/Predictions/{dataset}/"
#         "testResample2.csv"
#     )
#     assert os.path.exists(test_file)
#     _check_forecasting_file_format(
#         test_file, num_results_lines=DATA_TEST_SIZES[dataset]
#     )
#
#     # test present results checking
#     forecasting_experiments.run_experiment(args)
#
#     os.remove(test_file)
#
#
# def test_run_forecasting_experiment_main():
#     """Test forecasting experiments main with test data and forecaster."""
#     forecaster = "DummyForecaster"
#     dataset = "ShampooSales"
#
#     # run twice to test results present check
#     for _ in range(2):
#         runpy.run_path(
#             (
#                 "./tsml_eval/experiments/forecasting_experiments.py"
#                 if os.getcwd().split("\\")[-1] != "tests"
#                 else "../forecasting_experiments.py"
#             ),
#             run_name="__main__",
#         )
#
#     test_file = (
#         f"{_FORECASTER_RESULTS_PATH}{forecaster}/Predictions/{dataset}/"
#         "testResample0.csv"
#     )
#     assert os.path.exists(test_file)
#     _check_forecasting_file_format(test_file)
#
#     os.remove(test_file)
#
#
# def test_run_threaded_forecasting_experiment():
#     """Test threaded forecasting experiments with test data and forecaster."""
#     forecaster = "DummyForecaster"
#     dataset = "ShampooSales"
#
#     args = [
#         _TEST_DATA_PATH,
#         _FORECASTER_RESULTS_PATH,
#         forecaster,
#         dataset,
#         "1",
#         "-nj",
#         "2",
#         # also test transforms and benchmark time here
#         "--row_normalise",
#         "--data_transform_name",
#         "Padder",
#         "--benchmark_time",
#     ]
#
#     threaded_forecasting_experiments.run_experiment(args)
#
#     test_file = (
#         f"{_FORECASTER_RESULTS_PATH}{forecaster}/Predictions/{dataset}/"
#         "testResample1.csv"
#     )
#     assert os.path.exists(test_file)
#     _check_forecasting_file_format(test_file)
#
#     # test present results checking
#     threaded_forecasting_experiments.run_experiment(args)
#
#     # this covers the main method and experiment function result file checking
#     runpy.run_path(
#         (
#             "./tsml_eval/experiments/threaded_forecasting_experiments.py"
#             if os.getcwd().split("\\")[-1] != "tests"
#             else "../threaded_forecasting_experiments.py"
#         ),
#         run_name="__main__",
#     )
#
#     os.remove(test_file)
#
#
# def test_run_forecasting_experiment_invalid_estimator():
#     """Test run_forecasting_experiment method with invalid estimator."""
#     with pytest.raises(TypeError, match="forecaster must be a"):
#         run_forecasting_experiment(
#             [],
#             [],
#             DummyClassifier(),
#             [],
#         )
#
#
# def test_get_forecaster_by_name():
#     """Test get_forecaster_by_name method."""
#     forecaster_lists = [
#         set_forecaster.other_forecasters,
#     ]
#
#     forecaster_dict = {}
#     all_forecaster_names = []
#
#     for forecaster_list in forecaster_lists:
#         _check_set_method(
#             get_forecaster_by_name,
#             forecaster_list,
#             forecaster_dict,
#             all_forecaster_names,
#         )
#
#     _check_set_method_results(
#         forecaster_dict,
#         estimator_name="Forecasters",
#         method_name="get_forecaster_by_name",
#     )
#
#
# def test_get_forecaster_by_name_invalid():
#     """Test get_forecaster_by_name method with invalid estimator."""
#     with pytest.raises(ValueError, match="UNKNOWN FORECASTER"):
#         get_forecaster_by_name("invalid")
#
#
# def test_aeon_forecasters_available():
#     """Test all aeon forecasters are available."""
#     excluded = [
#         "RegressionForecaster"
#     ]
#
#     est = [e for e, _ in all_estimators(type_filter="forecaster")]
#     for e in est:
#         if e in excluded:
#             continue
#
#         try:
#             assert get_forecaster_by_name(e) is not None
#         except ModuleNotFoundError:
#             continue
