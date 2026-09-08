"""Tests that the simulator implements the paper's protocol as written.

Each test corresponds to a claim the protocol makes about the design, so that a
drift between code and protocol fails here rather than silently changing what the
experiment measures.
"""

import numpy as np
import pytest

from tsml_eval._wip.simulation._interval import order_templates
from tsml_eval._wip.simulation._protocol import (
    MECHANISMS,
    REFERENCE_KL,
    REFERENCE_LENGTH,
    SUPPORT_LENGTHS,
    condition_id,
    kl_matched_variance_ratio,
    protocol_conditions,
    simulate_protocol_problem,
    support_length_amplitude,
)


# --------------------------------------------------------------------------- #
# order: same values, different arrangement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("length", [16, 32, 63, 64, 128])
def test_order_templates_are_permutations(length):
    """The two order classes must differ only in arrangement."""
    t0, t1 = order_templates(length)

    np.testing.assert_allclose(np.sort(t0), np.sort(t1), atol=1e-12)
    np.testing.assert_allclose(t0.mean(), t1.mean(), atol=1e-12)
    np.testing.assert_allclose(t0.var(), t1.var(), atol=1e-12)
    np.testing.assert_allclose(t0.min(), t1.min(), atol=1e-12)
    np.testing.assert_allclose(t0.max(), t1.max(), atol=1e-12)

    # and they are genuinely differently ordered
    assert not np.allclose(t0, t1)


@pytest.mark.parametrize("length", [16, 64, 128])
def test_order_templates_are_zero_mean_unit_rms(length):
    t0, t1 = order_templates(length)
    for t in (t0, t1):
        np.testing.assert_allclose(t.mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.sqrt(np.mean(t**2)), 1.0, atol=1e-12)


def test_order_class_zero_is_ascending():
    t0, _ = order_templates(64)
    assert np.all(np.diff(t0) > 0)


# --------------------------------------------------------------------------- #
# level
# --------------------------------------------------------------------------- #
def test_level_shifts_the_interval_mean_by_class():
    """Interval values are N((2y - 1) a, 1), so the class means separate by 2a."""
    strength = 2.0
    X, y, params = simulate_protocol_problem(
        mechanism="level",
        strength=strength,
        n_cases=4000,
        random_state=0,
        return_params=True,
    )
    start, length = params["anchor"], params["length"]
    window = X[:, 0, start : start + length]
    m0 = window[y == 0].mean()
    m1 = window[y == 1].mean()
    np.testing.assert_allclose(m0, -strength, atol=0.05)
    np.testing.assert_allclose(m1, strength, atol=0.05)
    # variance is unchanged by the level mechanism
    np.testing.assert_allclose(window[y == 0].var(), 1.0, atol=0.05)


def test_level_at_zero_strength_carries_no_class_information():
    X, y, params = simulate_protocol_problem(
        mechanism="level", strength=0.0, n_cases=2000, random_state=1,
        return_params=True,
    )
    start, length = params["anchor"], params["length"]
    window = X[:, 0, start : start + length]
    np.testing.assert_allclose(window[y == 0].mean(), window[y == 1].mean(), atol=0.05)


# --------------------------------------------------------------------------- #
# alignment: three jitters plus uniform
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("jitter", [0, REFERENCE_LENGTH // 4, REFERENCE_LENGTH])
def test_jitter_bounds_the_displacement(jitter):
    """Starts sit within [anchor - J, anchor + J], and spread with J."""
    _, _, params = simulate_protocol_problem(
        mechanism="level", jitter=jitter, n_cases=2000, random_state=2,
        return_params=True,
    )
    offsets = params["starts"] - params["anchor"]
    assert offsets.min() >= -jitter
    assert offsets.max() <= jitter
    if jitter == 0:
        assert np.all(offsets == 0)
    else:
        # a bounded but genuinely graduated displacement
        assert offsets.min() < 0 < offsets.max()
        np.testing.assert_allclose(offsets.std(), jitter / np.sqrt(3.0), rtol=0.15)


def test_alignment_regimes_are_distinct():
    """J = 0, J = L/4, J = L and uniform give increasing displacement spread."""
    spreads = []
    for jitter in (0, REFERENCE_LENGTH // 4, REFERENCE_LENGTH):
        _, _, params = simulate_protocol_problem(
            mechanism="level", jitter=jitter, n_cases=2000, random_state=3,
            return_params=True,
        )
        spreads.append((params["starts"] - params["anchor"]).std())
    assert spreads[0] == 0
    assert spreads[0] < spreads[1] < spreads[2]

    _, _, uniform = simulate_protocol_problem(
        mechanism="level", uniform_location=True, n_cases=2000, random_state=3,
        return_params=True,
    )
    assert uniform["starts"].std() > spreads[2]


def test_uniform_location_covers_the_series():
    _, _, params = simulate_protocol_problem(
        mechanism="level", uniform_location=True, n_cases=4000, random_state=4,
        return_params=True,
    )
    starts = params["starts"]
    assert starts.min() < 20
    assert starts.max() > 512 - params["length"] - 20


# --------------------------------------------------------------------------- #
# energy matching
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("length", SUPPORT_LENGTHS)
def test_energy_matching_holds_across_the_support_sweep(length):
    """a_L^2 * L = 64 for every length in the sweep."""
    a = support_length_amplitude(length)
    np.testing.assert_allclose(a**2 * length, REFERENCE_LENGTH, rtol=1e-12)


def test_energy_matching_is_one_at_the_reference_length():
    np.testing.assert_allclose(support_length_amplitude(REFERENCE_LENGTH), 1.0)


# --------------------------------------------------------------------------- #
# KL matching
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("length", SUPPORT_LENGTHS)
def test_kl_matched_ratio_fixes_the_summed_divergence(length):
    """L (r + 1/r - 2) / 2 equals the reference value of 16 at every length."""
    r = kl_matched_variance_ratio(length)
    summed_kl = length * (r + 1.0 / r - 2.0) / 2.0
    np.testing.assert_allclose(summed_kl, REFERENCE_KL, rtol=1e-10)


def test_kl_matched_ratio_agrees_with_the_reference_cell():
    """At L = 64 the matched ratio is 2, which is 1 + a^2 at a = 1."""
    np.testing.assert_allclose(kl_matched_variance_ratio(REFERENCE_LENGTH), 2.0,
                               rtol=1e-12)


@pytest.mark.parametrize("length", SUPPORT_LENGTHS)
def test_scale_mechanism_realises_the_matched_ratio(length):
    X, y, params = simulate_protocol_problem(
        mechanism="scale", length=length, match_kl=True, n_cases=4000,
        random_state=5, return_params=True,
    )
    start = params["anchor"]
    window = X[:, 0, start : start + length]
    ratio = window[y == 1].var() / window[y == 0].var()
    np.testing.assert_allclose(ratio, params["variance_ratio"], rtol=0.1)


# --------------------------------------------------------------------------- #
# shuffle control
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_shuffle_preserves_values_and_destroys_order(mechanism):
    """Same seed, shuffle on and off: interval multiset kept, arrangement lost."""
    common = dict(
        mechanism=mechanism, n_cases=50, random_state=6, return_params=True
    )
    X_plain, y_plain, params = simulate_protocol_problem(
        shuffle_interval=False, **common
    )
    X_shuf, y_shuf, params_shuf = simulate_protocol_problem(
        shuffle_interval=True, **common
    )
    assert params["anchor"] == params_shuf["anchor"]
    np.testing.assert_array_equal(y_plain, y_shuf)

    start, length = params["anchor"], params["length"]
    window = slice(start, start + length)

    for i in range(len(y_plain)):
        a = X_plain[i, 0, window]
        b = X_shuf[i, 0, window]
        np.testing.assert_allclose(np.sort(a), np.sort(b), atol=1e-12)

    # at least some case has actually been rearranged
    assert any(
        not np.allclose(X_plain[i, 0, window], X_shuf[i, 0, window])
        for i in range(len(y_plain))
    )


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_shuffle_leaves_the_background_untouched(mechanism):
    common = dict(
        mechanism=mechanism, n_cases=50, random_state=7, return_params=True
    )
    X_plain, _, params = simulate_protocol_problem(shuffle_interval=False, **common)
    X_shuf, _, _ = simulate_protocol_problem(shuffle_interval=True, **common)

    start, length = params["anchor"], params["length"]
    mask = np.ones(params["series_length"], dtype=bool)
    mask[start : start + length] = False
    np.testing.assert_allclose(X_plain[:, 0, mask], X_shuf[:, 0, mask], atol=1e-12)


def test_shuffle_removes_the_order_signal_but_not_the_level_signal():
    """The control's purpose: order loses its class information, level keeps it."""
    start_kw = dict(n_cases=2000, random_state=8, return_params=True)

    X, y, params = simulate_protocol_problem(
        mechanism="order", shuffle_interval=True, **start_kw
    )
    window = X[:, 0, params["anchor"] : params["anchor"] + params["length"]]
    # arrangement destroyed, and order carries nothing else
    np.testing.assert_allclose(window[y == 0].mean(), window[y == 1].mean(), atol=0.05)
    np.testing.assert_allclose(window[y == 0].var(), window[y == 1].var(), atol=0.1)

    X, y, params = simulate_protocol_problem(
        mechanism="level", shuffle_interval=True, **start_kw
    )
    window = X[:, 0, params["anchor"] : params["anchor"] + params["length"]]
    # the histogram difference survives shuffling
    assert abs(window[y == 1].mean() - window[y == 0].mean()) > 1.0


# --------------------------------------------------------------------------- #
# condition grid
# --------------------------------------------------------------------------- #
def test_condition_grid_matches_the_protocol_count():
    conditions = protocol_conditions()
    assert len(conditions) == 58


def test_condition_ids_are_unique():
    conditions = protocol_conditions()
    ids = [condition_id(c) for c in conditions]
    assert len(set(ids)) == len(ids)


def test_every_condition_generates():
    for condition in protocol_conditions():
        X, y = simulate_protocol_problem(
            mechanism=condition["mechanism"],
            length=condition["length"],
            strength=condition["strength"],
            jitter=condition["jitter"],
            uniform_location=condition["uniform_location"],
            shuffle_interval=condition["shuffle_interval"],
            match_kl=condition["match_kl"],
            n_cases=20,
            random_state=9,
        )
        assert X.shape == (20, 1, 512)
        assert set(np.unique(y)) == {0, 1}
        assert np.isfinite(X).all()


def test_splits_share_the_population_anchor():
    """Train and test must place the informative region identically."""
    _, _, train = simulate_protocol_problem(
        mechanism="level", n_cases=100, random_state=10, return_params=True
    )
    _, _, test = simulate_protocol_problem(
        mechanism="level", n_cases=100, random_state=11,
        anchor=train["anchor"], return_params=True,
    )
    assert train["anchor"] == test["anchor"]
