"""Tests for the WIP ConvTran classifier."""

import numpy as np
import pytest

pytest.importorskip("torch")

from aeon.testing.data_generation import make_example_3d_numpy

from tsml_eval._wip.classification import ConvTranClassifier
from tsml_eval._wip.classification._convtran import _ConvTranNetwork
from tsml_eval.experiments import get_classifier_by_name


def test_convtran_experiment_lookup():
    """The experiment factory should expose both ConvTran names."""
    for name in ["ConvTran", "ConvTranClassifier"]:
        classifier = get_classifier_by_name(name, random_state=7, n_epochs=1)

        assert isinstance(classifier, ConvTranClassifier)
        assert classifier.random_state == 7
        assert classifier.n_epochs == 1


def test_convtran_aeon_shape_and_probabilities():
    """Fit with channels != timepoints and return valid probabilities."""
    X, y = make_example_3d_numpy(
        n_cases=20,
        n_channels=3,
        n_timepoints=12,
        n_labels=2,
        random_state=0,
    )
    y = np.where(y == 0, "class_a", "class_b")
    classifier = ConvTranClassifier(
        emb_size=8,
        dim_ff=16,
        num_heads=2,
        n_epochs=2,
        batch_size=4,
        validation_size=0.2,
        device="cpu",
        random_state=0,
    )

    classifier.fit(X, y)
    probabilities = classifier.predict_proba(X[:5])

    assert classifier.n_channels_ == 3
    assert classifier.n_timepoints_ == 12
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert set(classifier.predict(X[:5])).issubset(set(y))


def test_convtran_network_rejects_transposed_aeon_input():
    """Guard against silently swapping the channel and time axes."""
    network = _ConvTranNetwork(
        n_channels=3,
        n_timepoints=12,
        n_classes=2,
        emb_size=8,
        num_heads=2,
        dim_ff=16,
        dropout=0.01,
    )

    with pytest.raises(ValueError, match="input shape changed"):
        network(np.zeros((4, 12, 3), dtype=np.float32))


def test_convtran_is_repeatable_on_cpu():
    """The same seed should reproduce CPU probabilities."""
    X, y = make_example_3d_numpy(
        n_cases=16,
        n_channels=2,
        n_timepoints=10,
        n_labels=2,
        random_state=1,
    )
    parameters = {
        "emb_size": 8,
        "dim_ff": 16,
        "num_heads": 2,
        "n_epochs": 1,
        "batch_size": 4,
        "validation_size": 0.25,
        "device": "cpu",
        "random_state": 42,
    }

    first = ConvTranClassifier(**parameters).fit(X, y).predict_proba(X)
    second = ConvTranClassifier(**parameters).fit(X, y).predict_proba(X)

    np.testing.assert_allclose(first, second, atol=1e-7)
