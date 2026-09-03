"""Tests for the WIP TS2Vec regressor."""

import numpy as np
import pytest

pytest.importorskip("torch")

from tsml_eval._wip.regression import TS2VecRegressor
from tsml_eval.experiments import get_regressor_by_name


def _data(n_channels=2):
    rng = np.random.RandomState(12)
    X = rng.normal(size=(16, n_channels, 16)).astype(np.float32)
    y = (X[:, 0].mean(axis=1) - 0.5 * X[:, -1].mean(axis=1)).astype(
        np.float32
    )
    return X, y


def test_ts2vec_regressor_factory_lookup():
    """Expose both supported experiment names."""
    for name in ["TS2Vec", "TS2VecRegressor"]:
        regressor = get_regressor_by_name(name, random_state=7, n_iters=1)
        assert isinstance(regressor, TS2VecRegressor)
        assert regressor.random_state == 7


@pytest.mark.parametrize("n_channels", [1, 3])
def test_ts2vec_regressor_predictions(n_channels):
    """Return finite scalar predictions for univariate and multivariate data."""
    X, y = _data(n_channels=n_channels)
    regressor = TS2VecRegressor(
        output_dims=8,
        hidden_dims=8,
        depth=2,
        n_iters=1,
        batch_size=4,
        device="cpu",
        random_state=0,
    ).fit(X, y)
    predictions = regressor.predict(X[:4])

    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
