"""Tests for forecasting experiments."""

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
