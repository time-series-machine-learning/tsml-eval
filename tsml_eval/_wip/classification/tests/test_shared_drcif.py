"""Verification tests for SharedDrCIF and its shared interval transform."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.feature_based import Catch22
from aeon.utils.numba.stats import row_mean

from tsml_eval._wip.classification import SharedDrCIF, FastDrCIF
from tsml_eval._wip.classification._shared_interval_transform import (
    SharedIntervalTransform,
    drcif_random_intervals,
    dyadic_intervals,
)


def _data():
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=50, return_y=True, random_state=0
    )
    X2, _ = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=50, return_y=True, random_state=1
    )
    return X, y, X2


def test_dyadic_intervals():
    """Grid properties: full series first, tiled + half-shifted levels, no
    duplicates, respects min length and max depth."""
    ivs = dyadic_intervals(16, min_interval_length=3, max_depth=6)
    # depth 0: (0,16); depth 1: two tiles + one shift; depth 2: four tiles +
    # three shifts; depth 3 would be length 2 < 3 so stops
    assert ivs[0] == (0, 16)
    assert len(ivs) == 1 + 3 + 7
    assert len(set(ivs)) == len(ivs)
    assert all(e - s >= 3 for s, e in ivs)
    assert all(0 <= s < e <= 16 for s, e in ivs)

    # max_depth caps the grid even when lengths allow deeper levels
    assert len(dyadic_intervals(16, 3, 1)) == 1 + 3


def test_transform_values_match_direct_computation():
    """Spot check feature columns against direct Catch22/stat computation."""
    X, _, _ = _data()
    t = SharedIntervalTransform(max_depth=2)
    Xt = t.fit_transform(X)
    assert Xt.shape == (20, t.n_features_)
    assert Xt.dtype == np.float32

    # first block is the base representation, first interval (0, 50):
    # 22 catch22 columns then the 7 summary stats
    s, e = t.intervals_[0][0]
    sl = X[:, :, s:e]
    expected_c22 = Catch22(outlier_norm=True).fit_transform(sl)
    np.testing.assert_allclose(Xt[:, :22], expected_c22, rtol=1e-5)
    np.testing.assert_allclose(
        Xt[:, 22], row_mean(np.ascontiguousarray(sl[:, 0, :])), rtol=1e-5
    )


def test_transform_pool_sizes():
    """drcif29 gives 29 features per interval, union35 gives 35."""
    X, _, _ = _data()
    t29 = SharedIntervalTransform(features="drcif29", max_depth=2).fit(X)
    t35 = SharedIntervalTransform(features="union35", max_depth=2).fit(X)
    n_intervals = sum(len(ivs) for ivs in t29.intervals_)
    assert t29.n_features_ == n_intervals * 29
    assert t35.n_features_ == n_intervals * 35


def test_random_scheme_seeded():
    """Random intervals are reproducible and size-matched to the dyadic grid."""
    X, _, _ = _data()
    t1 = SharedIntervalTransform(
        interval_scheme="random", max_depth=2, random_state=3
    ).fit(X)
    t2 = SharedIntervalTransform(
        interval_scheme="random", max_depth=2, random_state=3
    ).fit(X)
    td = SharedIntervalTransform(interval_scheme="dyadic", max_depth=2).fit(X)
    assert t1.intervals_ == t2.intervals_
    assert [len(i) for i in t1.intervals_] == [len(i) for i in td.intervals_]


def test_drcif_random_intervals_respect_drcif_rule():
    """DrCIF random intervals obey min length and the 0.5 * m max length."""
    rng = np.random.RandomState(0)
    m = 100
    ivs = drcif_random_intervals(m, 500, rng, min_interval_length=3, max_interval_prop=0.5)
    assert len(ivs) == 500
    for s, e in ivs:
        assert 0 <= s < e <= m
        assert e - s >= 3
        assert e - s <= int(0.5 * m)


def test_shared_drcif2_matches_shared_drcif_feature_dim():
    """FastDrCIF (random) and SharedDrCIF (dyadic) share feature dimension
    and FastDrCIF fits, predicts and is deterministic at a fixed seed."""
    X, y, X2 = _data()
    d = SharedDrCIF(max_interval_depth=2, random_state=0).fit(X, y)
    r = FastDrCIF(max_interval_depth=2, random_state=0).fit(X, y)
    assert r._transformer.n_features_ == d._transformer.n_features_
    assert r.interval_scheme == "random"

    p1 = r.predict_proba(X2)
    p2 = FastDrCIF(max_interval_depth=2, random_state=0).fit(X, y).predict_proba(X2)
    assert np.array_equal(p1, p2)


def test_shared_drcif_unbagged_by_default():
    """Default deployment is unbagged (bootstrap off) and does not advertise a
    train estimate; test-only fit/predict, determinism and the fit time hook."""
    X, y, X2 = _data()

    clf = SharedDrCIF(max_interval_depth=2, random_state=0).fit(X, y)
    assert clf._estimator.bootstrap is False
    assert clf.get_tag("capability:train_estimate") is False
    assert clf.fit_time_millis_ >= 0

    p1 = clf.predict_proba(X2)
    assert p1.shape == (10, 2)
    assert np.allclose(p1.sum(axis=1), 1)

    p2 = (
        SharedDrCIF(max_interval_depth=2, random_state=0)
        .fit(X, y)
        .predict_proba(X2)
    )
    assert np.array_equal(p1, p2)


def test_shared_drcif_train_estimate_opt_in():
    """train_estimate=True re-enables bagging and OOB train estimates."""
    X, y, X2 = _data()

    clf = SharedDrCIF(max_interval_depth=2, train_estimate=True, random_state=0)
    train_probs = clf.fit_predict_proba(X, y)
    assert clf._estimator.bootstrap is True
    assert clf.get_tag("capability:train_estimate") is True
    assert train_probs.shape == (20, 2)
    assert np.allclose(train_probs.sum(axis=1), 1)
    assert np.isfinite(train_probs).all()


def test_banded_gates_features_by_length():
    """Banded mode computes only length-eligible catch22 features per interval,
    yields fewer columns than full, and stays deterministic."""
    from tsml_eval._wip.classification._shared_interval_transform import (
        CATCH22_LENGTH_THRESHOLDS,
        SharedIntervalTransform,
        feature_names,
    )

    X, y, X2 = _data()
    full = SharedIntervalTransform(interval_scheme="random", banded=False, random_state=0).fit(X)
    band = SharedIntervalTransform(interval_scheme="random", banded=True, random_state=0).fit(X)

    assert band.n_features_ < full.n_features_
    assert band.transform(X).shape[1] == band.n_features_

    # every interval's catch22 columns are exactly the length-eligible features
    per_iv = {}
    for r, iv, s, e, L, kind, name in band.column_meta_:
        if kind == "catch22":
            per_iv.setdefault((r, iv, L), []).append(name)
    for (r, iv, L), names in per_iv.items():
        assert names == [f for f in feature_names if L >= CATCH22_LENGTH_THRESHOLDS[f]]

    # a length-hungry feature never appears below its threshold
    for r, iv, s, e, L, kind, name in band.column_meta_:
        if name == "PD_PeriodicityWang_th0_01":
            assert L >= 50

    p1 = FastDrCIF(banded=True, max_interval_depth=3, random_state=0).fit(X, y).predict_proba(X2)
    p2 = FastDrCIF(banded=True, max_interval_depth=3, random_state=0).fit(X, y).predict_proba(X2)
    assert np.array_equal(p1, p2)


def test_shared_drcif_variants_fit():
    """union35 and random-interval variants build and predict."""
    X, y, X2 = _data()
    for kwargs in [
        {"features": "union35"},
        {"interval_scheme": "random"},
    ]:
        clf = SharedDrCIF(max_interval_depth=2, random_state=0, **kwargs)
        preds = clf.fit(X, y).predict(X2)
        assert preds.shape == (10,)
