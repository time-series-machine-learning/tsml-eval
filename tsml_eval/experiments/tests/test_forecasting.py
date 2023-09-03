import os

from tsml_eval.experiments import set_forecaster
from tsml_eval.experiments.forecasting_experiments import run_experiment
from tsml_eval.utils.test_utils import EXEMPT_ESTIMATOR_NAMES, _check_set_method
from tsml_eval.utils.tests.test_results_writing import _check_forecasting_file_format


def test_run_forecasting_experiment():
    """Test forecasting experiments with test data and forecasting."""
    forecaster = "NaiveForecaster"
    dataset = "ShampooSales"

    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../datasets/"
    )
    result_path = (
        "./test_output/forecasting/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/forecasting/"
    )

    args = [
        data_path,
        result_path,
        forecaster,
        dataset,
        "0",
        "-ow",
    ]

    run_experiment(args)

    test_file = f"{result_path}{forecaster}/Predictions/{dataset}/testResults.csv"

    assert os.path.exists(test_file)

    _check_forecasting_file_format(test_file)

    os.remove(test_file)


def test_set_forecasters():
    """Test set_forecasters method."""
    forecaster_lists = [
        set_forecaster.ml_forecasters,
        set_forecaster.other_forecasters,
    ]

    forecaster_dict = {}
    all_forecaster_names = []

    for forecaster_list in forecaster_lists:
        _check_set_method(
            set_forecaster.set_forecaster,
            forecaster_list,
            forecaster_dict,
            all_forecaster_names,
        )

    for estimator in EXEMPT_ESTIMATOR_NAMES:
        if estimator in forecaster_dict:
            forecaster_dict.pop(estimator)

    if not all(forecaster_dict.values()):
        missing_keys = [key for key, value in forecaster_dict.items() if not value]

        raise ValueError(
            "All forecasters seen in set_forecaster must have an entry for the full "
            "class name (usually with default parameters). forecasters with missing "
            f"entries: {missing_keys}."
        )
