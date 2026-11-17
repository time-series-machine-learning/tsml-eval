"""Tests for the band-pass and normalised classifier variants."""

import numpy as np
import pytest
from aeon.classification.compose import ClassifierPipeline
from aeon.transformations.collection import Normalizer

from tsml_eval.estimators.classification._eeg_preprocessing import BandPassFilter
from tsml_eval.experiments import get_classifier_by_name


def _sines(freq, n=6, length=400, sfreq=200.0):
    """Return n two-channel sines of one frequency."""
    t = np.arange(length) / sfreq
    return np.stack([np.stack([np.sin(2 * np.pi * freq * t)] * 2) for _ in range(n)])


def test_band_pass_keeps_the_band_and_removes_the_rest():
    """Pass a sine inside the band, remove one outside."""
    f = BandPassFilter(sfreq=200.0, l_freq=8.0, h_freq=30.0)
    inside = f.fit_transform(_sines(15))
    outside = f.fit_transform(_sines(60))
    assert inside.shape == (6, 2, 400)
    assert inside[:, :, 50:-50].std() > 0.9 * _sines(15).std()
    assert outside[:, :, 50:-50].std() < 0.05  # edges carry the filter transient


def test_band_pass_short_series_and_bad_cut_offs():
    """Filter series shorter than the pad; reject cut-offs above Nyquist."""
    assert BandPassFilter(100.0, 1.0, 40.0).fit_transform(
        _sines(10, length=20)
    ).shape == (6, 2, 20)
    with pytest.raises(ValueError, match="h_freq < sfreq / 2"):
        BandPassFilter(50.0, 8.0, 30.0).fit_transform(_sines(10))


@pytest.mark.parametrize(
    "name, low, high", [("ROCKET-BP830", 8.0, 30.0), ("ROCKET-BP140", 1.0, 40.0)]
)
def test_band_pass_names(name, low, high):
    """Build band-pass pipelines from names; require sfreq."""
    c = get_classifier_by_name(name, random_state=0, sfreq=160)
    assert isinstance(c, ClassifierPipeline)
    f = c.transformers[0]
    assert isinstance(f, BandPassFilter)
    assert (f.sfreq, f.l_freq, f.h_freq) == (160.0, low, high)
    assert type(c.classifier).__name__ == "RocketClassifier"
    with pytest.raises(ValueError, match="sfreq"):
        get_classifier_by_name(name, random_state=0)


def test_norm_name_and_names_with_dashes_unchanged():
    """Build -norm pipelines; leave other dashed names alone."""
    c = get_classifier_by_name("ROCKET-norm", random_state=0)
    assert isinstance(c.transformers[0], Normalizer)
    assert type(get_classifier_by_name("STC-aeon")).__name__ == (
        "ShapeletTransformClassifier"
    )


def test_band_passed_classifier_fits():
    """Fit and predict with a band-passed classifier."""
    X = np.concatenate([_sines(12), _sines(25)])
    X = X + np.random.default_rng(0).normal(scale=0.2, size=X.shape)
    y = np.array([0] * 6 + [1] * 6)
    c = get_classifier_by_name("ROCKET-BP830", random_state=0, sfreq=200.0)
    c.fit(X, y)
    assert c.predict(X).shape == (12,)
