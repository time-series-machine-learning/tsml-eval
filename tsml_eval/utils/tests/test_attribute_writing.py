import os

from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.sklearn import RotationForestClassifier
from tsml.datasets import load_minimal_chinatown

from tsml_eval.testing.test_utils import _TEST_OUTPUT_PATH
from tsml_eval.utils.experiments import estimator_attributes_to_file


def test_estimator_attributes_to_file():
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
