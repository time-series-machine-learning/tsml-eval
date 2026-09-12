"""Tests for the corrected Arsenal member weighting."""

import numpy as np
import pytest
from aeon.classification.convolution_based import Arsenal
from aeon.datasets import load_arrow_head, load_italy_power_demand
from numpy.testing import assert_array_almost_equal

from tsml_eval._wip.arsenal_weighting._arsenal_fixed import (
    _AEON_SUPPORTED,
    FixedWeightArsenal,
)

pytestmark = pytest.mark.skipif(
    not _AEON_SUPPORTED,
    reason="installed aeon's Arsenal predates the member weighting this fix targets",
)

_PARAMS = {"n_kernels": 500, "n_estimators": 8, "random_state": 0}


def test_binary_weights_are_not_degenerate():
    """The released Arsenal weights every binary member 1.0; the fix must not."""
    X, y = load_italy_power_demand(split="train")
    assert len(np.unique(y)) == 2

    released = Arsenal(**_PARAMS).fit(X, y)
    fixed = FixedWeightArsenal(**_PARAMS).fit(X, y)

    # The defect being fixed: every member scores exactly 1.0 regardless of the data.
    assert np.allclose(released.weights_, 1.0)
    assert not np.allclose(fixed.weights_, 1.0)
    assert all(0.0 < w <= 1.0 for w in fixed.weights_)
    assert set(fixed.weighting_paths_) == {"cv-binary"}


def test_multiclass_is_unchanged():
    """More than two classes keeps the fast path, so behaviour must be identical."""
    X, y = load_arrow_head(split="train")
    assert len(np.unique(y)) > 2

    released = Arsenal(**_PARAMS).fit(X, y)
    fixed = FixedWeightArsenal(**_PARAMS).fit(X, y)

    assert_array_almost_equal(np.asarray(released.weights_), np.asarray(fixed.weights_))
    assert set(fixed.weighting_paths_) == {"gcv"}

    X_test, _ = load_arrow_head(split="test")
    assert_array_almost_equal(released.predict_proba(X_test), fixed.predict_proba(X_test))


def test_predictions_remain_valid():
    """Probabilities still form a distribution after reweighting."""
    X, y = load_italy_power_demand(split="train")
    X_test, _ = load_italy_power_demand(split="test")
    probas = FixedWeightArsenal(**_PARAMS).fit(X, y).predict_proba(X_test)

    assert probas.shape == (len(X_test), len(np.unique(y)))
    assert_array_almost_equal(probas.sum(axis=1), 1.0)
