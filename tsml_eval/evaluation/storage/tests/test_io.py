import os

from tsml_eval.evaluation.storage.classifier_results import ClassifierResults
from tsml_eval.evaluation.storage.clusterer_results import ClustererResults
from tsml_eval.evaluation.storage.forecaster_results import ForecasterResults
from tsml_eval.evaluation.storage.regressor_results import RegressorResults
from tsml_eval.testing.test_utils import _TEST_OUTPUT_PATH, _TEST_RESULTS_PATH
from tsml_eval.utils.validation import validate_results_file


def test_classifier_results():
    """Test ClassifierResults loading and saving."""
    cr = ClassifierResults().load_from_file(
        _TEST_RESULTS_PATH
        + "/classification/ROCKET/Predictions/MinimalChinatown/testResample0.csv"
    )
    cr.save_to_file(_TEST_OUTPUT_PATH + "/classification/results_io/")

    assert validate_results_file(
        _TEST_OUTPUT_PATH + "/classification/results_io/testResample0.csv"
    )

    os.remove(_TEST_OUTPUT_PATH + "/classification/results_io/testResample0.csv")


def test_clusterer_results():
    """Test ClustererResults loading and saving."""
    cr = ClustererResults().load_from_file(
        _TEST_RESULTS_PATH
        + "/clustering/KMeans/Predictions/MinimalChinatown/trainResample0.csv"
    )
    cr.save_to_file(_TEST_OUTPUT_PATH + "/clustering/results_io/")

    assert validate_results_file(
        _TEST_OUTPUT_PATH + "/clustering/results_io/trainResample0.csv"
    )

    os.remove(_TEST_OUTPUT_PATH + "/clustering/results_io/trainResample0.csv")


def test_regressor_results():
    """Test RegressorResults loading and saving."""
    cr = RegressorResults().load_from_file(
        _TEST_RESULTS_PATH
        + "/regression/ROCKET/Predictions/MinimalGasPrices/testResample0.csv"
    )
    cr.save_to_file(_TEST_OUTPUT_PATH + "/regression/results_io/")

    assert validate_results_file(
        _TEST_OUTPUT_PATH + "/regression/results_io/testResample0.csv"
    )

    os.remove(_TEST_OUTPUT_PATH + "/regression/results_io/testResample0.csv")


def test_forecaster_results():
    """Test ForecasterResults loading and saving."""
    cr = ForecasterResults().load_from_file(
        _TEST_RESULTS_PATH
        + "/forecasting/NaiveForecaster/Predictions/ShampooSales/testResample0.csv"
    )
    cr.save_to_file(_TEST_OUTPUT_PATH + "/forecasting/results_io/")

    assert validate_results_file(
        _TEST_OUTPUT_PATH + "/forecasting/results_io/testResample0.csv"
    )

    os.remove(_TEST_OUTPUT_PATH + "/forecasting/results_io/testResample0.csv")
