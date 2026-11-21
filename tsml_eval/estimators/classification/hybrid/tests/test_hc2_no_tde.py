"""Tests for the in-memory HIVE-COTE 2 without TDE."""

import numpy as np
from aeon.testing.data_generation import make_example_3d_numpy

from tsml_eval.estimators.classification.hybrid import HIVECOTEV2NoTDE


def test_hc2_no_tde_weights_and_prior_correction():
    """CAWPE weights from train estimates, then division by the prior to the beta."""
    X, y = make_example_3d_numpy(
        n_cases=24, n_channels=2, n_timepoints=30, n_labels=3, random_state=0
    )
    params = HIVECOTEV2NoTDE._get_test_params()
    plain = HIVECOTEV2NoTDE(**params, prior_correction=0, random_state=0).fit(X, y)
    corrected = HIVECOTEV2NoTDE(**params, prior_correction=0.5, random_state=0)
    corrected.fit(X, y)

    assert len(plain.weights_) == 3 and all(0 <= w <= 1 for w in plain.weights_)
    p0 = plain.predict_proba(X)
    dists = sum(
        w * e.predict_proba(X) for w, e in zip(plain.weights_, plain.estimators_)
    )
    np.testing.assert_allclose(p0, dists / dists.sum(axis=1, keepdims=True))

    expected = p0 / corrected.prior_**0.5
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(corrected.predict_proba(X), expected)
    assert set(corrected.predict(X)) <= set(np.unique(y))
