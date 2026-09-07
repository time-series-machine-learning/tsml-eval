"""Tests for the FIRE / SharedDrCIF regressors."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy

from tsml_eval._wip.regression import FIRERegressor, SharedDrCIFRegressor


def _data(n_cases=30, n_timepoints=60, seed=0):
    return make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=1,
        n_timepoints=n_timepoints,
        regression_target=True,
        random_state=seed,
    )


@pytest.mark.parametrize("cls", [SharedDrCIFRegressor, FIRERegressor])
def test_fit_predict_shape(cls):
    X, y = _data()
    Xte, _ = _data(n_cases=10, seed=1)
    reg = cls(max_interval_depth=2, random_state=0).fit(X, y)
    p = reg.predict(Xte)
    assert p.shape == (10,)
    assert np.isfinite(p).all()
    assert reg.n_features_used_ > 0
    assert reg.fit_time_millis_ >= 0


def test_seeded_deterministic():
    X, y = _data()
    Xte, _ = _data(n_cases=10, seed=1)
    p1 = FIRERegressor(max_interval_depth=2, random_state=42).fit(X, y).predict(Xte)
    p2 = FIRERegressor(max_interval_depth=2, random_state=42).fit(X, y).predict(Xte)
    np.testing.assert_allclose(p1, p2)


def test_ensemble_averages_heads():
    """The two-head FIRE prediction is the mean of its single-head predictions."""
    X, y = _data()
    Xte, _ = _data(n_cases=10, seed=1)
    kw = dict(max_interval_depth=2, random_state=7)

    et = FIRERegressor(heads=("extratrees",), **kw).fit(X, y).predict(Xte)
    rid = FIRERegressor(heads=("ridge",), **kw).fit(X, y).predict(Xte)
    both = FIRERegressor(heads=("extratrees", "ridge"), **kw).fit(X, y).predict(Xte)

    np.testing.assert_allclose(both, (et + rid) / 2, rtol=1e-5)


def test_single_head_matches_shared_drcif_minimal():
    """FIRE(extratrees only) == SharedDrCIFRegressor with the same config."""
    X, y = _data()
    Xte, _ = _data(n_cases=10, seed=1)
    kw = dict(max_interval_depth=2, random_state=3)

    fire = FIRERegressor(heads=("extratrees",), **kw).fit(X, y).predict(Xte)
    shared = (
        SharedDrCIFRegressor(
            features="minimal", banded=True, interval_scheme="random", **kw
        )
        .fit(X, y)
        .predict(Xte)
    )
    np.testing.assert_allclose(fire, shared, rtol=1e-5)


def test_train_estimate_opt_in():
    X, y = _data()
    reg = FIRERegressor(max_interval_depth=2, train_estimate=True, random_state=0)
    assert reg.get_tag("capability:train_estimate")
    preds = reg.fit_predict(X, y)
    assert preds.shape == (len(y),)
    assert np.isfinite(preds).all()


def test_train_estimate_off_by_default():
    reg = FIRERegressor(max_interval_depth=2)
    assert not reg.get_tag("capability:train_estimate")


def test_invalid_head_raises():
    X, y = _data()
    with pytest.raises(ValueError, match="Unknown head"):
        FIRERegressor(heads=("nonsense",), max_interval_depth=2).fit(X, y)
