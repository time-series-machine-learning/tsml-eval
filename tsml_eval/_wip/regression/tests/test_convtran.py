"""Tests for the WIP ConvTran regressor."""

import numpy as np
import pytest

pytest.importorskip("torch")

from tsml_eval._wip.regression import ConvTranRegressor
from tsml_eval.experiments import get_regressor_by_name


def _make_regression_data(n_cases=20, n_channels=3, n_timepoints=12, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_cases, n_channels, n_timepoints)).astype(np.float32)
    y = (2.0 * X[:, 0].mean(axis=1) - X[:, -1].mean(axis=1)).astype(
        np.float32
    )
    return X, y


def _small_regressor(random_state=0):
    return ConvTranRegressor(
        emb_size=8,
        dim_ff=16,
        num_heads=2,
        n_epochs=2,
        batch_size=4,
        validation_size=0.2,
        device="cpu",
        random_state=random_state,
    )


def test_convtran_regressor_experiment_lookup():
    """The experiment factory should expose both ConvTran regressor names."""
    for name in ["ConvTran", "ConvTranRegressor"]:
        regressor = get_regressor_by_name(name, random_state=7, n_epochs=1)

        assert isinstance(regressor, ConvTranRegressor)
        assert regressor.random_state == 7
        assert regressor.n_epochs == 1


def test_convtran_regressor_aeon_shape_and_predictions():
    """Fit multivariate aeon input and return finite scalar predictions."""
    X, y = _make_regression_data()
    regressor = _small_regressor().fit(X, y)
    predictions = regressor.predict(X[:5])

    assert regressor.n_channels_ == 3
    assert regressor.n_timepoints_ == 12
    assert regressor.model_.out.out_features == 1
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()


def test_convtran_regressor_accepts_univariate_input():
    """Treat univariate input as the one-channel special case."""
    X, y = _make_regression_data(n_channels=1)
    predictions = _small_regressor().fit(X, y).predict(X[:3])

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()


def test_convtran_regressor_is_repeatable_on_cpu():
    """The same seed should reproduce CPU predictions."""
    X, y = _make_regression_data(n_cases=16, n_channels=2, n_timepoints=10, seed=1)

    first = _small_regressor(random_state=42).fit(X, y).predict(X)
    second = _small_regressor(random_state=42).fit(X, y).predict(X)

    np.testing.assert_allclose(first, second, atol=1e-7)
