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
    s, e, d = t.intervals_[0][0]
    sl = X[:, :, s:e:d]
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
    for s, e, d in ivs:
        assert d == 1  # dilation off by default
        assert 0 <= s < e <= m
        assert e - s >= 3
        assert e - s <= int(0.5 * m)


def test_dilation_scales_to_interval_length():
    """With dilation on, each interval's point count stays within the length
    cap while its dilated span can reach the whole series; dilation is >=1 and
    the span always fits."""
    rng = np.random.RandomState(0)
    m = 200
    ivs = drcif_random_intervals(
        m, 2000, rng, min_interval_length=3, max_interval_prop=0.5, dilation=True
    )
    saw_dilated = False
    for s, e, d in ivs:
        assert d >= 1
        assert 0 <= s < e <= m  # dilated span fits the series
        L = len(range(s, e, d))  # point count
        assert 3 <= L <= int(0.5 * m)
        if d > 1:
            saw_dilated = True
            # span expands with dilation: (L-1)*d + 1
            assert e - s == (L - 1) * d + 1
    assert saw_dilated  # geometric draw should produce some dilation > 1

    # d_max scales with interval length: a half-length interval maxes at d=2
    X, y, X2 = _data()
    from tsml_eval._wip.classification import FastDrCIF
    p1 = FastDrCIF(dilation=True, max_interval_depth=3, random_state=0).fit(X, y).predict_proba(X2)
    p2 = FastDrCIF(dilation=True, max_interval_depth=3, random_state=0).fit(X, y).predict_proba(X2)
    assert np.array_equal(p1, p2)  # deterministic


def test_family_configuration():
    """The three-class family: SharedDrCIF (random, no gates, no dilation),
    FastDrCIF (random + gates), FastDrCIF_D (random + gates + dilation), all
    with the constant-feature filter on."""
    from tsml_eval._wip.classification import FastDrCIF_D

    X, y, X2 = _data()
    s = SharedDrCIF(max_interval_depth=2, random_state=0).fit(X, y)
    f = FastDrCIF(max_interval_depth=2, random_state=0).fit(X, y)
    fd = FastDrCIF_D(max_interval_depth=2, random_state=0).fit(X, y)

    # all random intervals, all with the constant filter
    for clf in (s, f, fd):
        assert clf.interval_scheme == "random"
        assert clf.drop_constant is True

    # gating: SharedDrCIF is unbanded, Fast* are banded
    assert s.banded is False and f.banded is True and fd.banded is True
    # dilation: only FastDrCIF_D
    assert s.dilation is False and f.dilation is False and fd.dilation is True

    # banding gives FastDrCIF fewer transform columns than SharedDrCIF
    assert f._transformer.n_features_ < s._transformer.n_features_
    # dilation adds no columns per interval: unbanded, dilation on vs off gives
    # the same count (in banded mode the drawn intervals differ, so counts can
    # differ, but dilation itself never adds columns)
    s_dil = SharedDrCIF(
        max_interval_depth=2, dilation=True, random_state=0
    ).fit(X, y)
    assert s_dil._transformer.n_features_ == s._transformer.n_features_

    # all fit, predict and are deterministic
    for clf, ctor in ((s, SharedDrCIF), (f, FastDrCIF), (fd, FastDrCIF_D)):
        p2 = ctor(max_interval_depth=2, random_state=0).fit(X, y).predict_proba(X2)
        assert np.array_equal(clf.predict_proba(X2), p2)


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


def test_representations_configurable():
    """Default is the 3 DrCIF representations; dropping periodogram and adding
    second-order differences change the interval-set structure and still fit."""
    from tsml_eval._wip.classification._shared_interval_transform import (
        SharedIntervalTransform,
    )

    X, y, X2 = _data()

    default = SharedIntervalTransform(interval_scheme="random", random_state=0).fit(X)
    nopgram = SharedIntervalTransform(
        interval_scheme="random", representations=("base", "diff1"), random_state=0
    ).fit(X)
    diff2 = SharedIntervalTransform(
        interval_scheme="random",
        representations=("base", "diff1", "diff2"),
        random_state=0,
    ).fit(X)

    assert len(default.intervals_) == 3
    assert len(nopgram.intervals_) == 2  # periodogram dropped
    assert len(diff2.intervals_) == 3
    assert nopgram._periodogram is None
    # invalid representation rejected
    with pytest.raises(ValueError, match="Unknown representation"):
        SharedIntervalTransform(representations=("base", "bogus")).fit(X)

    # classifiers with these representations fit and predict
    for reps in [("base", "diff1"), ("base", "diff1", "diff2")]:
        clf = SharedDrCIF(
            representations=reps, max_interval_depth=3, random_state=0
        ).fit(X, y)
        assert clf.predict(X2).shape == (10,)


def test_drop_constant_features():
    """Constant/all-zero columns are dropped before the forest, the kept mask
    is applied at predict, and toggling the filter changes the count used."""
    X, y, X2 = _data()

    clf = SharedDrCIF(max_interval_depth=3, drop_constant=True, random_state=0).fit(X, y)
    Xt = clf._transformer.transform(X)
    n_constant = int((Xt.std(axis=0) == 0).sum())
    assert clf.n_features_used_ == Xt.shape[1] - n_constant
    assert clf._estimator.n_features_in_ == clf.n_features_used_
    # predict still works with the mask applied
    assert clf.predict(X2).shape == (10,)

    # with the filter off, all columns are kept
    clf2 = SharedDrCIF(max_interval_depth=3, drop_constant=False, random_state=0).fit(X, y)
    assert clf2.n_features_used_ == Xt.shape[1]


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
