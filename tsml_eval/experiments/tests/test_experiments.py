"""General tests for the experiments module."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
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


def test_device_description_without_a_deep_learning_framework(monkeypatch):
    """The device should be reported as CPU when no framework is loaded."""
    from tsml_eval.experiments.experiments import _device_description

    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert _device_description() == "CPU"
    # Reporting the device must not import a framework, as that would add seconds
    # of start-up and a large amount of memory to every non-deep experiment.
    assert "tensorflow" not in sys.modules
    assert "torch" not in sys.modules


def test_device_description_reports_a_tensorflow_gpu_model(monkeypatch):
    """A loaded framework reporting a GPU should name the device in the results."""
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

    monkeypatch.setitem(sys.modules, "tensorflow", fake)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    description = _device_description()

    assert description == "NVIDIA A100  80GB"
    assert "," not in description


def _fake_torch(cuda_available, device_names=(), cuda_error=None, mps=False):
    """Construct a minimal fake torch module for device-reporting tests."""

    def is_available():
        if cuda_error is not None:
            raise cuda_error
        return cuda_available

    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=is_available,
            device_count=lambda: len(device_names),
            get_device_name=lambda index: device_names[index],
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
    )


def _fake_tensorflow(device_names=(), error=None):
    """Construct a minimal fake TensorFlow module for device-reporting tests."""

    gpus = [
        SimpleNamespace(name=f"/physical_device:GPU:{i}")
        for i in range(len(device_names))
    ]

    def list_physical_devices(kind):
        if error is not None:
            raise error
        return gpus if kind == "GPU" else []

    details_by_id = {id(gpu): name for gpu, name in zip(gpus, device_names)}
    return SimpleNamespace(
        config=SimpleNamespace(
            list_physical_devices=list_physical_devices,
            experimental=SimpleNamespace(
                get_device_details=lambda gpu: {
                    "device_name": details_by_id[id(gpu)]
                }
            ),
        )
    )


def test_device_description_torch_without_cuda(monkeypatch):
    """A loaded PyTorch without an accelerator should report CPU."""
    from tsml_eval.experiments.experiments import _device_description

    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))

    assert _device_description() == "CPU"


def test_device_description_reports_one_torch_gpu(monkeypatch):
    """A loaded PyTorch should report its CUDA device name."""
    from tsml_eval.experiments.experiments import _device_description

    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(True, ("NVIDIA H200 NVL",))
    )

    assert _device_description() == "NVIDIA H200 NVL"


def test_device_description_reports_multiple_torch_gpus_comma_free(monkeypatch):
    """All CUDA devices should be named without introducing CSV fields."""
    from tsml_eval.experiments.experiments import _device_description

    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(True, ("NVIDIA A100, 80GB", "NVIDIA H200 NVL")),
    )

    description = _device_description()
    assert description == "NVIDIA A100  80GB; NVIDIA H200 NVL"
    assert "," not in description


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
def test_device_description_framework_errors_do_not_propagate(monkeypatch, framework):
    """Framework device-query failures should not prevent result writing."""
    from tsml_eval.experiments.experiments import _device_description

    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    error = RuntimeError("device query failed")
    fake = (
        _fake_tensorflow(error=error)
        if framework == "tensorflow"
        else _fake_torch(False, cuda_error=error)
    )
    monkeypatch.setitem(sys.modules, framework, fake)

    assert _device_description() == "unknown"


def test_device_description_reports_devices_from_both_frameworks(monkeypatch):
    """Devices from both loaded frameworks are reported and duplicate names removed."""
    from tsml_eval.experiments.experiments import _device_description

    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        _fake_tensorflow(("Shared GPU", "TensorFlow GPU")),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(True, ("Shared GPU", "PyTorch GPU")),
    )

    assert _device_description() == "Shared GPU; TensorFlow GPU; PyTorch GPU"


def test_non_deep_experiment_does_not_import_frameworks(monkeypatch, tmp_path):
    """A CPU experiment should record CPU without importing a deep framework."""
    from tsml_eval.experiments.experiments import run_classification_experiment

    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    X_train = np.arange(48, dtype=float).reshape(4, 1, 12)
    X_test = np.arange(24, dtype=float).reshape(2, 1, 12)
    y_train = np.array([0, 1, 0, 1])
    y_test = np.array([0, 1])

    run_classification_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        DummyClassifier(),
        str(tmp_path),
        classifier_name="Dummy",
        dataset_name="Toy",
        resample_id=0,
        benchmark_time=False,
    )

    result_file = tmp_path / "Dummy" / "Predictions" / "Toy" / "testResample0.csv"
    _check_classification_file_format(str(result_file), num_results_lines=2)
    assert "Device: CPU." in result_file.read_text().splitlines()[0]
    assert "tensorflow" not in sys.modules
    assert "torch" not in sys.modules
