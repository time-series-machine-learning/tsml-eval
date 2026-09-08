"""The paper's focused explanatory simulation protocol.

This module is the executable form of the protocol in
``interval_explanatory_study.tex``. The protocol is authoritative: where it and
the earlier ``simulate_interval_shape_data`` simulator disagree, this module
follows the protocol.

The generator here differs from that simulator in the ways the protocol requires:
a single informative interval rather than three, series of length 512, an
explicit strength ``a`` rather than an amplitude/noise ratio, graduated
displacement rather than aligned-or-random placement, the two mechanisms
``level`` and ``order`` alongside ``scale`` and ``frequency``, and the two
matching rules that stop the support-length sweep from confounding length with
signal strength.

Mechanisms, inside an interval of length ``L`` with strength ``a``, on a
background of independent standard Gaussian noise:

* ``level``      observations are ``N((2y - 1) a, 1)``
* ``scale``      observations are ``N(0, 1 + y a^2)``
* ``order``      a template ``a t_y`` plus ``N(0, 1)``, where ``t_1`` is a
                 permutation of ``t_0``, so the classes share every order
                 statistic and differ only in arrangement
* ``frequency``  ``sqrt(2) a sin(2 pi f_y i / L + phi)`` plus ``N(0, 1)``, with
                 ``f_0 = 2``, ``f_1 = 4`` cycles per interval and ``phi`` drawn
                 per case

Placement. A population anchor is drawn once per replicate and shared by the
training and test cases. Each case is displaced from it by an offset drawn
uniformly from the integers ``[-J, J]``, independently of class. ``J = 0`` is
the aligned condition. A separate ``uniform`` regime draws each start uniformly
over every valid position instead.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "MECHANISMS",
    "REFERENCE_LENGTH",
    "REFERENCE_KL",
    "support_length_amplitude",
    "kl_matched_variance_ratio",
    "scale_variance_ratio",
    "simulate_protocol_problem",
    "protocol_conditions",
    "condition_id",
]

import numpy as np
from sklearn.utils import check_random_state

from tsml_eval._wip.simulation._interval import order_templates

#: The four mechanisms, in protocol order.
MECHANISMS = ("level", "scale", "order", "frequency")

#: Series length used throughout the protocol.
SERIES_LENGTH = 512

#: The reference cell of the design: L = 64 at a = 1.
REFERENCE_LENGTH = 64

#: Summed directed KL divergence at the reference cell, which the scale
#: mechanism holds constant across support lengths.
REFERENCE_KL = 16.0

#: Cycles per interval for the frequency mechanism, per class.
FREQUENCIES = (2, 4)


# --------------------------------------------------------------------------- #
# Matching rules
# --------------------------------------------------------------------------- #
def support_length_amplitude(length, reference_length=REFERENCE_LENGTH):
    """Strength that keeps total signal energy constant as the support grows.

    ``a_L = sqrt(64 / L)``, so ``a_L**2 * L`` is the same for every length in the
    sweep. Without this a longer interval would carry more signal simply by being
    longer, and the support-length sweep would confound length with strength.
    """
    if length < 1:
        raise ValueError("length must be positive.")
    return np.sqrt(reference_length / length)


def kl_matched_variance_ratio(length, reference_kl=REFERENCE_KL):
    """Class-one interval variance that fixes the summed KL divergence.

    For ``L`` independent observations, the two directed Kullback--Leibler
    divergences between ``N(0, 1)`` and ``N(0, r)`` sum to
    ``L (r + 1/r - 2) / 2``. Writing ``z = r + 1/r``, holding that sum at
    ``reference_kl`` gives ``z = 2 + 2 * reference_kl / L`` and

        r = (z + sqrt(z^2 - 4)) / 2,

    the larger of the two roots. At the reference cell, ``L = 64`` and
    ``reference_kl = 16`` give ``z = 2.5`` and ``r = 2``, which coincides with
    ``1 + a^2`` at ``a = 1``: the matched scale sweep agrees with the plain
    strength convention where the two overlap.
    """
    if length < 1:
        raise ValueError("length must be positive.")
    z = 2.0 + 2.0 * reference_kl / length
    return (z + np.sqrt(z * z - 4.0)) / 2.0


def scale_variance_ratio(length, strength, match_kl):
    """Class-one interval variance for the scale mechanism.

    With ``match_kl`` the ratio comes from :func:`kl_matched_variance_ratio`, so
    distributional separation rather than signal energy is held constant.
    Otherwise it is ``1 + a^2`` directly from the mechanism definition.
    """
    if match_kl:
        return kl_matched_variance_ratio(length)
    return 1.0 + strength**2


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def _shuffle_seed(random_state):
    """A stream distinct from the generator's, derived from the same seed."""
    if isinstance(random_state, (int, np.integer)):
        return int(random_state) + 104729  # a fixed prime offset
    return random_state


def _interval_template_and_sigma(mechanism, label, length, strength, rng, ratio):
    """Template and per-point noise sd inside the interval for one case."""
    if mechanism == "level":
        return np.full(length, (2 * label - 1) * strength, dtype=float), 1.0
    if mechanism == "scale":
        return np.zeros(length, dtype=float), np.sqrt(ratio if label == 1 else 1.0)
    if mechanism == "order":
        return strength * order_templates(length)[label], 1.0
    if mechanism == "frequency":
        phase = rng.uniform(0.0, 2.0 * np.pi)
        i = np.arange(length)
        signal = np.sqrt(2.0) * strength * np.sin(
            2.0 * np.pi * FREQUENCIES[label] * i / length + phase
        )
        return signal, 1.0
    raise ValueError(f"Unknown mechanism '{mechanism}', valid: {MECHANISMS}.")


def _draw_starts(n_cases, series_length, length, anchor, jitter, uniform, rng):
    """Interval start for every case."""
    if uniform:
        return rng.randint(0, series_length - length + 1, size=n_cases)
    if jitter == 0:
        return np.full(n_cases, anchor, dtype=int)
    offsets = rng.randint(-jitter, jitter + 1, size=n_cases)
    return anchor + offsets


def draw_anchor(series_length, length, max_jitter, rng):
    """Population anchor, with enough margin for the largest bounded jitter.

    Drawn once per replicate and shared by the training and test cases, so the
    informative region sits in the same place in both.
    """
    low = max_jitter
    high = series_length - length - max_jitter
    if high < low:
        raise ValueError(
            f"an interval of length {length} with jitter {max_jitter} does not "
            f"fit in a series of length {series_length}."
        )
    return int(rng.randint(low, high + 1))


def simulate_protocol_problem(
    mechanism="level",
    length=REFERENCE_LENGTH,
    strength=1.0,
    jitter=0,
    uniform_location=False,
    shuffle_interval=False,
    match_kl=False,
    n_cases=200,
    series_length=SERIES_LENGTH,
    anchor=None,
    random_state=None,
    return_params=False,
):
    """Generate one protocol condition.

    Parameters
    ----------
    mechanism : str, default="level"
        One of ``level``, ``scale``, ``order``, ``frequency``.
    length : int, default=64
        Length ``L`` of the single informative interval.
    strength : float, default=1.0
        The strength ``a``.
    jitter : int, default=0
        Case starts are displaced from the anchor by an offset drawn uniformly
        from the integers ``[-jitter, jitter]``. Zero is the aligned condition.
        Ignored when ``uniform_location`` is True.
    uniform_location : bool, default=False
        Draw each start uniformly over every valid position instead.
    shuffle_interval : bool, default=False
        Permute the observed values inside the interval, per case, after the
        noise has been added. Location and every value are preserved and only
        the arrangement is destroyed.
    match_kl : bool, default=False
        For the scale mechanism, take the variance ratio from the KL-matched
        rule rather than from ``1 + a^2``.
    n_cases : int, default=200
        Number of cases, split exactly evenly between the two classes.
    anchor : int or None, default=None
        Population anchor. Pass the training split's anchor when generating the
        test split so the two share the informative region.

    Returns
    -------
    X : np.ndarray of shape (n_cases, 1, series_length)
    y : np.ndarray of shape (n_cases,)
    params : dict, only if ``return_params``
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"Unknown mechanism '{mechanism}', valid: {MECHANISMS}.")
    if n_cases % 2:
        raise ValueError("n_cases must be even so the classes balance exactly.")
    rng = check_random_state(random_state)

    if anchor is None:
        anchor = draw_anchor(series_length, length, jitter, rng)

    ratio = scale_variance_ratio(length, strength, match_kl)

    y = np.repeat([0, 1], n_cases // 2)
    starts = _draw_starts(
        n_cases, series_length, length, anchor, jitter, uniform_location, rng
    )

    X = rng.normal(0.0, 1.0, size=(n_cases, series_length))
    for i in range(n_cases):
        start = starts[i]
        template, sigma = _interval_template_and_sigma(
            mechanism, y[i], length, strength, rng, ratio
        )
        window = slice(start, start + length)
        # The background draw is reused as the interval's unit-variance noise,
        # scaled where the mechanism calls for it, so that changing only the
        # mechanism changes only the signal.
        X[i, window] = template + X[i, window] * sigma

    # The shuffle control must be a paired diagnostic: with the same seed, the
    # shuffled and unshuffled versions of a condition have to be the same data
    # with only the within-interval arrangement destroyed. Drawing the
    # permutations from the main stream would consume variates and desynchronise
    # every later case, so the shuffle gets a stream of its own.
    if shuffle_interval:
        shuffle_rng = check_random_state(
            None if random_state is None else _shuffle_seed(random_state)
        )
        for i in range(n_cases):
            window = slice(starts[i], starts[i] + length)
            X[i, window] = shuffle_rng.permutation(X[i, window])

    order = rng.permutation(n_cases)
    X, y, starts = X[order], y[order], starts[order]
    X = X[:, np.newaxis, :]

    if return_params:
        return X, y, {
            "mechanism": mechanism,
            "length": length,
            "strength": strength,
            "jitter": jitter,
            "uniform_location": uniform_location,
            "shuffle_interval": shuffle_interval,
            "match_kl": match_kl,
            "anchor": anchor,
            "starts": starts,
            "variance_ratio": ratio if mechanism == "scale" else None,
            "series_length": series_length,
        }
    return X, y


# --------------------------------------------------------------------------- #
# Condition grid
# --------------------------------------------------------------------------- #
SUPPORT_LENGTHS = (16, 32, 64, 128)
STRENGTHS = (0.5, 1.0, 2.0)


def condition_id(condition):
    """Stable, filesystem-safe name for a condition, used as the dataset name."""
    bits = [
        condition["mechanism"],
        "L%d" % condition["length"],
        "a%g" % condition["strength"],
    ]
    if condition["uniform_location"]:
        bits.append("uniform")
    else:
        bits.append("J%d" % condition["jitter"])
    if condition["shuffle_interval"]:
        bits.append("shuffled")
    if condition["match_kl"]:
        bits.append("klmatched")
    return "-".join(bits).replace(".", "p")


def protocol_conditions():
    """Every unique condition in the protocol design.

    Built from the four sweeps rather than hard-coded, so the count is derived
    and can be reconciled against the protocol's stated total. Conditions shared
    between sweeps appear once: the support-length sweep's ``L = 64`` cells
    coincide with the alignment sweep's aligned and uniform cells because
    ``a_64 = 1``, and the strength sweep's ``a = 1`` cell coincides with the
    aligned cell.
    """
    conditions = []
    seen = set()

    def add(**kwargs):
        condition = {
            "mechanism": kwargs["mechanism"],
            "length": kwargs.get("length", REFERENCE_LENGTH),
            "strength": kwargs.get("strength", 1.0),
            "jitter": kwargs.get("jitter", 0),
            "uniform_location": kwargs.get("uniform_location", False),
            "shuffle_interval": kwargs.get("shuffle_interval", False),
            "match_kl": kwargs.get("match_kl", False),
            "sweep": kwargs["sweep"],
        }
        key = condition_id(condition)
        if key in seen:
            return
        seen.add(key)
        condition["id"] = key
        conditions.append(condition)

    # Alignment sweep: L = 64, a = 1, graduated displacement then uniform.
    for mechanism in MECHANISMS:
        for jitter in (0, REFERENCE_LENGTH // 4, REFERENCE_LENGTH):
            add(mechanism=mechanism, jitter=jitter, sweep="alignment")
        add(mechanism=mechanism, uniform_location=True, sweep="alignment")

    # Support-length sweep, energy or KL matched, in both placement regimes.
    for mechanism in MECHANISMS:
        for length in SUPPORT_LENGTHS:
            strength = float(support_length_amplitude(length))
            match_kl = mechanism == "scale" and length != REFERENCE_LENGTH
            for uniform in (False, True):
                add(
                    mechanism=mechanism,
                    length=length,
                    strength=strength,
                    uniform_location=uniform,
                    match_kl=match_kl,
                    sweep="support-length",
                )

    # Strength sweep, aligned, at the reference length.
    for mechanism in MECHANISMS:
        for strength in STRENGTHS:
            add(mechanism=mechanism, strength=strength, sweep="strength")

    # Within-interval shuffle control, both placement regimes.
    for mechanism in MECHANISMS:
        for uniform in (False, True):
            add(
                mechanism=mechanism,
                uniform_location=uniform,
                shuffle_interval=True,
                sweep="shuffle",
            )

    # Zero-strength anchors: no class information, chance accuracy 0.5.
    for uniform in (False, True):
        add(
            mechanism="level",
            strength=0.0,
            uniform_location=uniform,
            sweep="zero-strength",
        )

    return conditions
