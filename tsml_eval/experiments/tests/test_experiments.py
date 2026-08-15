"""General tests for the experiments module."""

import os

from aeon.classification import DummyClassifier

from tsml_eval.experiments import (
    classification_experiments,
    load_and_run_classification_experiment,
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


def test_device_description_without_a_deep_learning_framework():
    """The device should be reported as CPU when no framework is loaded."""
    import sys

    from tsml_eval.experiments.experiments import _device_description

    tensorflow = sys.modules.pop("tensorflow", None)
    try:
        assert _device_description() == "CPU"
        # Reporting the device must not import a framework, as that would add seconds
        # of start-up and a large amount of memory to every non-deep experiment.
        assert "tensorflow" not in sys.modules
    finally:
        if tensorflow is not None:
            sys.modules["tensorflow"] = tensorflow


def test_device_description_reports_a_gpu_model():
    """A loaded framework reporting a GPU should name the device in the results."""
    import sys
    from types import SimpleNamespace

    from tsml_eval.experiments.experiments import _device_description

    gpu = SimpleNamespace(name="/physical_device:GPU:0")
    fake = SimpleNamespace(
        config=SimpleNamespace(
            list_physical_devices=lambda kind: [gpu] if kind == "GPU" else [],
            experimental=SimpleNamespace(
                # A comma here would be read as a field separator in the first line.
                get_device_details=lambda device: {"device_name": "NVIDIA A100, 80GB"}
            ),
        )
    )

    original = sys.modules.get("tensorflow")
    sys.modules["tensorflow"] = fake
    try:
        description = _device_description()
    finally:
        if original is None:
            del sys.modules["tensorflow"]
        else:
            sys.modules["tensorflow"] = original

    assert description == "NVIDIA A100  80GB"
    assert "," not in description
