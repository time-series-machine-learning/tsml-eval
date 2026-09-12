"""Tests for building HIVE-COTE from results files without the data."""

import numpy as np
from aeon.datasets import load_italy_power_demand
from numpy.testing import assert_array_almost_equal

from tsml_eval.estimators.classification.hybrid.hivecote_from_file import (
    FromFileHIVECOTE,
)
from tsml_eval.estimators.classification.hybrid.hivecote_from_results import (
    build_hivecote_from_results,
)
from tsml_eval.evaluation.storage import ClassifierResults
from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH

_COMPONENTS = [
    _TEST_RESULTS_PATH + "/classification/Arsenal",
    _TEST_RESULTS_PATH + "/classification/DrCIF",
    _TEST_RESULTS_PATH + "/classification/STC",
    _TEST_RESULTS_PATH + "/classification/TDE",
]


def test_build_hivecote_from_results(tmp_path):
    """Building from results matches FromFileHIVECOTE without loading the data."""
    test_results, train_results = build_hivecote_from_results(
        _COMPONENTS,
        "ItalyPowerDemand",
        0,
        str(tmp_path),
        classifier_name="HC2",
        write_train_file=True,
    )

    # The expected values are those asserted in test_hivecote_from_file.
    assert test_results.probabilities.shape[1] == 2
    assert_array_almost_equal(
        test_results.probabilities[0], np.array([0.0785, 0.9215]), decimal=4
    )
    assert train_results is not None

    # Identical to running the classifier with the data loaded.
    X_train, y_train = load_italy_power_demand(split="train")
    X_test, _ = load_italy_power_demand(split="test")
    hc2 = FromFileHIVECOTE(
        classifiers=[c + "/Predictions/ItalyPowerDemand/" for c in _COMPONENTS],
        random_state=0,
    )
    hc2.fit(X_train, y_train)
    assert_array_almost_equal(test_results.probabilities, hc2.predict_proba(X_test))
    # Results files store label indices, so map the classifier's labels back.
    indices = np.searchsorted(hc2.classes_, hc2.predict(X_test))
    assert_array_almost_equal(np.asarray(test_results.predictions), indices)
    assert test_results.fit_time == hc2.fit_time_millis_


def test_written_results_are_readable(tmp_path):
    """The written files load back as valid results with matching statistics."""
    test_results, _ = build_hivecote_from_results(
        _COMPONENTS, "ItalyPowerDemand", 0, str(tmp_path), classifier_name="HC2"
    )

    loaded = ClassifierResults().load_from_file(
        str(tmp_path / "HC2" / "Predictions" / "ItalyPowerDemand" / "testResample0.csv")
    )
    assert loaded.accuracy == test_results.accuracy
    assert loaded.n_cases == test_results.n_cases
    assert_array_almost_equal(loaded.probabilities, test_results.probabilities)


def test_new_weights(tmp_path):
    """Passing weights replaces the accuracy weights."""
    equal, _ = build_hivecote_from_results(
        _COMPONENTS,
        "ItalyPowerDemand",
        0,
        str(tmp_path),
        classifier_name="HC2-Equal",
        new_weights=[1, 1, 1, 1],
    )
    accuracy_weighted, _ = build_hivecote_from_results(
        _COMPONENTS, "ItalyPowerDemand", 0, str(tmp_path), classifier_name="HC2"
    )
    assert not np.allclose(equal.probabilities, accuracy_weighted.probabilities)


def test_prior_correction(tmp_path):
    """Prior correction changes the decision rule and beta=0 is a no-op."""
    base, _ = build_hivecote_from_results(
        _COMPONENTS, "ItalyPowerDemand", 0, str(tmp_path), classifier_name="HC2"
    )
    zero, _ = build_hivecote_from_results(
        _COMPONENTS, "ItalyPowerDemand", 0, str(tmp_path),
        classifier_name="HC2-b0", prior_correction=0.0,
    )
    half, _ = build_hivecote_from_results(
        _COMPONENTS, "ItalyPowerDemand", 0, str(tmp_path),
        classifier_name="HC2-b05", prior_correction=0.5,
    )

    assert_array_almost_equal(base.probabilities, zero.probabilities)
    assert not np.allclose(base.probabilities, half.probabilities)
    # Probabilities are still a valid distribution after correction.
    assert_array_almost_equal(half.probabilities.sum(axis=1), 1.0)
    # The written predictions agree with the corrected probabilities.
    assert np.array_equal(
        np.asarray(half.predictions), np.argmax(half.probabilities, axis=1)
    )
