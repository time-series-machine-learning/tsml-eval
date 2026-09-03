"""Tests for the WIP TimesNet regressor."""

import numpy as np
import pytest

pytest.importorskip("torch")

from tsml_eval._wip.regression import TimesNetRegressor
from tsml_eval.experiments import get_regressor_by_name


def _data(n_channels=2):
    rng = np.random.RandomState(10)
    X = rng.normal(size=(16, n_channels, 12)).astype(np.float32)
    y = (X[:, 0].mean(axis=1) + 0.5 * X[:, -1].mean(axis=1)).astype(
        np.float32
    )
    return X, y


def _regressor():
    return TimesNetRegressor(
        e_layers=1,
        d_model=8,
        d_ff=8,
        top_k=2,
        num_kernels=2,
        n_epochs=1,
        batch_size=4,
        validation_size=0.25,
        device="cpu",
        random_state=0,
    )


def test_timesnet_regressor_factory_lookup():
    """Expose both supported experiment names."""
    for name in ["TimesNet", "TimesNetRegressor"]:
        regressor = get_regressor_by_name(name, random_state=7, n_epochs=1)
        assert isinstance(regressor, TimesNetRegressor)
        assert regressor.random_state == 7


@pytest.mark.parametrize("n_channels", [1, 3])
def test_timesnet_regressor_predictions(n_channels):
    """Return finite scalar predictions for univariate and multivariate data."""
    X, y = _data(n_channels=n_channels)
    regressor = _regressor().fit(X, y)
    predictions = regressor.predict(X[:4])

    assert regressor.network_.projection.out_features == 1
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
