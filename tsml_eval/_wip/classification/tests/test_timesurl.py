"""Tests for the WIP TimesURL classifier."""

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aeon.testing.data_generation import make_example_3d_numpy

from tsml_eval._wip.classification import TimesURLClassifier
from tsml_eval.experiments import get_classifier_by_name


def test_timesurl_experiment_lookup():
    """The experiment factory should expose both TimesURL names."""
    for name in ["TimesURL", "TimesURLClassifier"]:
        classifier = get_classifier_by_name(name, random_state=7, n_iters=1)

        assert isinstance(classifier, TimesURLClassifier)
        assert classifier.random_state == 7
        assert classifier.n_iters == 1


def test_timesurl_defaults_match_original_classification_command():
    """Defaults should reproduce the authors' train.py classification setup."""
    classifier = TimesURLClassifier()

    assert classifier.output_dims == 320
    assert classifier.hidden_dims == 64
    assert classifier.depth == 10
    assert classifier.n_iters is None
    assert classifier.n_epochs is None
    assert classifier.batch_size == 8
    assert classifier.learning_rate == 1e-4
    assert classifier.max_train_length == 3000
    assert classifier.temperature == 1.0
    assert classifier.lmd == 0.01
    assert classifier.segment_num == 3
    assert classifier.mask_ratio_per_seg == 0.05
    assert classifier.eval_protocol == "svm"


def test_timesurl_shape_probabilities_and_train_only_scaling(capsys):
    """Fit NCT data, return valid probabilities, and retain train statistics."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=2,
        n_timepoints=8,
        n_labels=2,
        random_state=0,
    )
    y = np.where(y == 0, "class_a", "class_b")
    classifier = TimesURLClassifier(
        output_dims=8,
        hidden_dims=8,
        depth=2,
        n_iters=2,
        batch_size=4,
        device="cpu",
        random_state=0,
    )

    classifier.fit(X, y)
    expected_mean = np.transpose(X, (0, 2, 1)).reshape(-1, 2).mean(axis=0)
    np.testing.assert_allclose(classifier.scaler_.mean_, expected_mean)
    training_mean = classifier.scaler_.mean_.copy()

    probabilities = classifier.predict_proba(X[:3] + 10_000)

    assert classifier.n_channels_ == 2
    assert classifier.n_timepoints_ == 8
    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
    np.testing.assert_array_equal(classifier.scaler_.mean_, training_mean)
    assert capsys.readouterr().out == ""


def test_timesurl_auto_device_falls_back_to_cpu(monkeypatch):
    """Auto device selection must not leave the invalid literal 'auto'."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        monkeypatch.setattr(mps, "is_available", lambda: False)

    assert TimesURLClassifier()._resolve_device(torch) == "cpu"


def test_timesurl_uses_package_imports_without_path_mutation():
    """Loading the adapter should not install the vendored directory on sys.path."""
    assert not any(path.endswith("_timesurl_original") for path in sys.path)
