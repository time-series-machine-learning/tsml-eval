"""Test experiment utilities."""

import os

import numpy as np
import pytest
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.sklearn import RotationForestClassifier

from tsml_eval.datasets._loaders import load_minimal_chinatown
from tsml_eval.testing.testing_utils import _TEST_OUTPUT_PATH
from tsml_eval.utils.experiments import (
    _check_existing_results,
    _results_present,
    estimator_attributes_to_file,
    timing_benchmark,
)
from tsml_eval.utils.results_writing import write_regression_results


@pytest.mark.parametrize("split", ["BOTH", "TRAIN", "TEST", None, "invalid"])
def test_results_present_split_inputs(split):
    """Test _results_present function with valid and invalid split inputs."""
    if split == "invalid":
        with pytest.raises(ValueError, match="Unknown split value"):
            _results_present(
                "test_output",
                "test",
                "test",
                split=split,
            )
    else:
        assert not _results_present(
            "test_output",
            "test",
            "test",
            split=split,
        )


@pytest.mark.parametrize("dataset_name", ["Data", "N/A"])
@pytest.mark.parametrize("resample_id", [None, 0])
def test_check_existing_results_finds_written_files(
    tmp_path, dataset_name, resample_id
):
    """Test _check_existing_results finds files where the results writer puts them."""
    write_regression_results(
        np.arange(3.0),
        np.arange(3.0),
        "Est",
        dataset_name,
        str(tmp_path),
        full_path=False,
        split="TEST",
        resample_id=resample_id,
    )
    assert _check_existing_results(
        str(tmp_path), "Est", dataset_name, resample_id, False, True, True
    ) == (False, True)


def test_timing_benchmark_invalid_input():
    """Test timing_benchmark function with invalid input."""
    with pytest.raises(ValueError):
        timing_benchmark(random_state="invalid")


def test_estimator_attributes_to_file():
    """Test writing estimator attributes to file."""
    estimator = ShapeletTransformClassifier(
        n_shapelet_samples=50,
        estimator=RotationForestClassifier(n_estimators=2),
    )
    X, y = load_minimal_chinatown()
    estimator.fit(X, y)

    test_dir = _TEST_OUTPUT_PATH + "/attribute_writing/"
    estimator_attributes_to_file(estimator, test_dir)

    assert os.path.exists(test_dir + "ShapeletTransformClassifier.txt")
    assert os.path.exists(test_dir + "estimator/estimator.txt")
    assert os.path.exists(test_dir + "_estimator/_estimator.txt")
    assert os.path.exists(test_dir + "_estimator/_base_estimator/_base_estimator.txt")
    assert os.path.exists(test_dir + "_estimator/_pcas_0_0/_pcas_0_0.txt")


def test_max_depth():
    """Test writing estimator attributes to file max depth parameter."""
    estimator = ShapeletTransformClassifier(
        n_shapelet_samples=50,
        estimator=RotationForestClassifier(n_estimators=2),
    )
    X, y = load_minimal_chinatown()
    estimator.fit(X, y)

    test_dir = _TEST_OUTPUT_PATH + "/attribute_writing_max_depth/"
    estimator_attributes_to_file(estimator, test_dir, max_depth=1)

    assert os.path.exists(test_dir + "ShapeletTransformClassifier.txt")
    assert os.path.exists(test_dir + "estimator/estimator.txt")
    assert os.path.exists(test_dir + "_estimator/_estimator.txt")
    assert not os.path.exists(
        test_dir + "_estimator/_base_estimator/_base_estimator.txt"
    )
    assert not os.path.exists(test_dir + "_estimator/_pcas_0_0/_pcas_0_0.txt")


def test_max_list_shape():
    """Test writing estimator attributes to file max list shape parameter."""
    estimator = ShapeletTransformClassifier(
        n_shapelet_samples=50,
        estimator=RotationForestClassifier(n_estimators=2),
    )
    X, y = load_minimal_chinatown()
    estimator.fit(X, y)

    test_dir = _TEST_OUTPUT_PATH + "/attribute_writing_max_list_shape/"
    estimator_attributes_to_file(estimator, test_dir, max_list_shape=1)

    assert os.path.exists(test_dir + "ShapeletTransformClassifier.txt")
    assert os.path.exists(test_dir + "estimator/estimator.txt")
    assert os.path.exists(test_dir + "_estimator/_estimator.txt")
    assert os.path.exists(test_dir + "_estimator/_base_estimator/_base_estimator.txt")
    assert not os.path.exists(test_dir + "_estimator/_pcas_0_0/_pcas_0_0.txt")
