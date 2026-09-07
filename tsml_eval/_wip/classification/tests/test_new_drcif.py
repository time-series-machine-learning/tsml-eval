"""Verification tests for NewDrCIF, QuantDrCIF and the quantile features."""

import warnings

import numpy as np
import pytest
from aeon.classification.interval_based import DrCIFClassifier
from aeon.testing.data_generation import make_example_3d_numpy

from tsml_eval._wip.classification import NewDrCIF, QuantDrCIF
from tsml_eval._wip.classification._quantile_stats import (
    row_quantile_10,
    row_quantile_25,
    row_quantile_25_centred,
    row_quantile_75,
    row_quantile_75_centred,
    row_quantile_90,
)

QUANTILE_FUNCS = [
    (row_quantile_10, 0.10, False),
    (row_quantile_25, 0.25, False),
    (row_quantile_75, 0.75, False),
    (row_quantile_90, 0.90, False),
    (row_quantile_25_centred, 0.25, True),
    (row_quantile_75_centred, 0.75, True),
]


def _data():
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=30, return_y=True, random_state=0
    )
    X2, _ = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=30, return_y=True, random_state=1
    )
    return X, y, X2


@pytest.mark.parametrize("func,q,centred", QUANTILE_FUNCS)
def test_quantile_functions_match_numpy(func, q, centred):
    """Each quantile feature must agree with np.quantile row-wise."""
    rng = np.random.RandomState(0)
    for X in [
        rng.normal(size=(20, 37)),
        rng.normal(size=(5, 1)),  # length-1 intervals
        np.full((6, 10), 3.7),  # constant intervals
        rng.uniform(-100, 100, size=(10, 4)),
    ]:
        expected = np.quantile(X, q, axis=1)
        if centred:
            expected = expected - X.mean(axis=1)
        np.testing.assert_allclose(func(X), expected, rtol=1e-12, atol=1e-12)


def test_new_drcif_matches_aeon_drcif():
    """Regression anchor: NewDrCIF (the 29-attribute pool clone) must be
    bit-identical to aeon's DrCIFClassifier at the same seed."""
    X, y, X2 = _data()

    probs_aeon = (
        DrCIFClassifier(n_estimators=5, random_state=42).fit(X, y).predict_proba(X2)
    )
    probs_new = NewDrCIF(n_estimators=5, random_state=42).fit(X, y).predict_proba(X2)
    assert np.array_equal(probs_aeon, probs_new)


def test_quant_drcif_same_seed_determinism():
    """Repeat fits with the same seed must produce identical output."""
    X, y, X2 = _data()

    p1 = QuantDrCIF(n_estimators=5, random_state=7).fit(X, y).predict_proba(X2)
    p2 = QuantDrCIF(n_estimators=5, random_state=7).fit(X, y).predict_proba(X2)
    assert np.array_equal(p1, p2)


def test_quant_drcif_n_jobs_invariance():
    """n_jobs=1 and n_jobs=2 must produce identical output at the same seed."""
    X, y, X2 = _data()

    p1 = (
        QuantDrCIF(n_estimators=5, random_state=7, n_jobs=1)
        .fit(X, y)
        .predict_proba(X2)
    )
    p2 = (
        QuantDrCIF(n_estimators=5, random_state=7, n_jobs=2)
        .fit(X, y)
        .predict_proba(X2)
    )
    assert np.array_equal(p1, p2)


def test_quant_drcif_pool_size_is_35():
    """The extended pool must contain exactly 35 attributes (22 catch22 + 7
    summary stats + 6 quantiles).

    BaseIntervalForest warns when att_subsample_size exceeds the pool, so
    requesting 35 must be silent and 36 must warn.
    """
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=20, return_y=True, random_state=0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        QuantDrCIF(n_estimators=1, att_subsample_size=35, random_state=0).fit(X, y)

    with pytest.warns(UserWarning, match="larger than the number of attributes"):
        QuantDrCIF(n_estimators=1, att_subsample_size=36, random_state=0).fit(X, y)
