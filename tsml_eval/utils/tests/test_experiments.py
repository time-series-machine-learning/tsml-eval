"""Test experiment utilities."""

import os

import pytest
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.sklearn import RotationForestClassifier
from tsml.datasets import load_minimal_chinatown

from tsml_eval.testing.testing_utils import _TEST_OUTPUT_PATH
from tsml_eval.utils.experiments import (
    _results_present,
    estimator_attributes_to_file,
    timing_benchmark,
)


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
