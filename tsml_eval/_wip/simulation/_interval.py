"""Interval-optimal data simulator.

A numpy redesign of the Java ``IntervalModel`` / ``SimulateIntervalData``
(tsml-java ``statistics.simulators``). It generates a classification problem
where interval-based classifiers should be (near) optimal.

Generative principle (what makes it *interval*-optimal)
-------------------------------------------------------
The discriminative signal is a fixed set of contiguous regions ("intervals")
that are:

* **shared across all classes** -- every class has signal in the *same* interval
  positions, so the intervals are phase aligned across the whole dataset;
* **filled with a class-specific shape** -- the classes differ only in *which
  shape* occupies those intervals (a sine, a step, a spike, ...);
* **surrounded by noise** -- everything outside the intervals is i.i.d. Gaussian
  noise carrying no class information.

Because the class information lives in fixed, aligned regions and is expressed
as the *within-region pattern* (i.e. captured by summary statistics over those
regions), a classifier that computes features on the right sub-intervals -- TSF,
CIF, DrCIF, QUANT, ... -- can separate the classes, while:

* whole-series/global-feature methods dilute the localised signal across the
  full length (most of which is noise);
* phase-invariant methods (shapelets, dictionary, convolution) pay for a
  location tolerance that is unnecessary here, since the discriminative regions
  never move -- only their *content* differs between classes.

This mirrors the "when are interval methods effective?" hypothesis: localised
class information in approximately aligned regions with noise elsewhere.

Design notes vs the Java original
---------------------------------
* **RNG.** The Java version relies on global static singletons (``Model.rand``
  and a per-model error seed of ``count * (seed + 1)``), which makes exact
  reproduction fragile. Here randomness is a single explicitly-threaded
  ``numpy`` generator (``random_state``), so a seed fully determines the output.
* **Vectorised.** Java generates value-by-value with a per-timestep linear scan
  for the active interval. Here each class has one clean signal *template* built
  once, and every case is ``template + Gaussian noise`` -- a couple of numpy ops.
* **K classes.** The Java simulator is hard-wired to two classes; this
  generalises to any number, each assigned a distinct shape.
* **Shapes** follow the Java intent (sine/triangle/step/spike/head-and-
  shoulders) but with standardised, consistently-scaled definitions rather than
  the Java boundary handling.
* **Interval placement** uses an exact non-overlapping construction (random
  gaps via stars-and-bars) instead of the Java midpoint rejection loop, which
  its own comments flag as imperfect.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["simulate_interval_shape_data", "SHAPES"]

import numpy as np
from sklearn.utils import check_random_state


# --------------------------------------------------------------------------- #
# Shapes: each returns a length-``length`` signal spanning roughly [base, base +
# amplitude]. The shapes are chosen to differ in the summary statistics interval
# classifiers compute (mean, slope, variance, extrema, ...), so that regional
# features separate the classes.
# --------------------------------------------------------------------------- #
def _sine(length, amplitude, base):
    """One full sinusoid cycle, oscillating about ``base`` (~zero mean shape)."""
    i = np.arange(length)
    return base + (amplitude / 2.0) * np.sin(2 * np.pi * i / max(length - 1, 1))


def _triangle(length, amplitude, base):
    """Symmetric tent: ``base`` at the ends rising to ``base + amplitude`` centre."""
    i = np.arange(length)
    mid = (length - 1) / 2.0
    return base + amplitude * (1.0 - np.abs(i - mid) / max(mid, 1))


def _step(length, amplitude, base):
    """A single level change: ``base`` then ``base + amplitude`` at the midpoint."""
    i = np.arange(length)
    return np.where(i < length // 2, base, base + amplitude).astype(float)


def _spike(length, amplitude, base):
    """Flat ``base`` with a narrow triangular spike to ``base + amplitude``."""
    i = np.arange(length)
    centre = (length - 1) / 2.0
    half = max(length // 8, 1)  # spike half-width
    val = base + amplitude * (1.0 - np.abs(i - centre) / half)
    return np.maximum(val, base)


def _head_and_shoulders(length, amplitude, base):
    """Three humps (small, large, small) via a sinusoid over thirds."""
    i = np.arange(length)
    third = length / 3.0
    val = np.full(length, base, dtype=float)
    left = i < third
    right = i >= 2 * third
    mid = ~left & ~right
    val[left] = base + (amplitude / 2.0) * np.sin(np.pi * i[left] / max(third, 1))
    val[right] = base + (amplitude / 2.0) * np.sin(
        np.pi * (i[right] - 2 * third) / max(third, 1)
    )
    val[mid] = base + amplitude * np.sin(np.pi * (i[mid] - third) / max(third, 1))
    return val


#: Registry of available shape generators (name -> callable).
SHAPES = {
    "sine": _sine,
    "triangle": _triangle,
    "step": _step,
    "spike": _spike,
    "head_shoulders": _head_and_shoulders,
}


def _place_intervals(series_length, n_intervals, interval_length, rng):
    """Random non-overlapping interval start points, sorted.

    Uses a stars-and-bars construction: draw ``n_intervals`` positions in the
    available slack, sort, then offset the i-th by ``i * interval_length`` so the
    intervals cannot overlap while the gaps between them are random.
    """
    slack = series_length - n_intervals * interval_length
    if slack < 0:
        raise ValueError(
            f"{n_intervals} intervals of length {interval_length} do not fit in "
            f"a series of length {series_length}."
        )
    starts = np.sort(rng.randint(0, slack + 1, size=n_intervals))
    starts = starts + np.arange(n_intervals) * interval_length
    return starts


def simulate_interval_shape_data(
    n_cases_per_class=(50, 50),
    series_length=500,
    n_intervals=3,
    interval_length=None,
    noise_to_signal=4,
    discriminator="shape",
    phase_alignment="fixed",
    shapes=None,
    amplitude=2.0,
    base=-1.0,
    noise_sigma=1.0,
    interval_scales=None,
    frequencies=None,
    random_state=None,
    return_params=False,
):
    """Generate an interval-optimal classification problem.

    All classes share the same interval positions; each class fills them with a
    distinct shape; noise is added everywhere. See the module docstring for why
    interval classifiers should be optimal on this data.

    Parameters
    ----------
    n_cases_per_class : sequence of int, default=(50, 50)
        Number of cases for each class; its length is the number of classes.
    series_length : int, default=500
        Length of every series.
    n_intervals : int, default=3
        Number of discriminative intervals (shared across classes).
    interval_length : int or None, default=None
        Length of each interval. If None, derived from ``noise_to_signal`` as
        ``series_length // (n_intervals * noise_to_signal)``.
    noise_to_signal : int, default=4
        Only used when ``interval_length`` is None; larger values give shorter
        intervals (less signal, more surrounding noise) -- the difficulty dial.
    discriminator : "shape", "scale" or "trend", default="shape"
        What distinguishes the classes within the shared intervals.

        * ``"shape"`` -- a class-specific deterministic *shape* (sine/step/...).
          Detectable by shape methods too (shapelets, convolution/PULSAR), so
          this is interval-*friendly* but not interval-*exclusive*.
        * ``"scale"`` -- the intervals are bursts of class-specific *variance*
          with no deterministic pattern. There is nothing for a shape detector
          to match, but an interval variance/spread statistic separates the
          classes cleanly -- designed to confound shapelet/convolutional methods
          while remaining easy for interval statistics.
        * ``"trend"`` -- the intervals contain a linear ramp of class-specific
          slope: a whole-interval statistic (slope) rather than a local shape.
        * ``"frequency"`` -- the intervals contain a sinusoid at a class-specific
          frequency (same amplitude, random phase per case). Mean/variance/slope
          are identical across classes, so purely time-domain interval features
          are blind; only spectral representations (periodogram / Fourier /
          autocorrelation) separate the classes. Probes the representation axis.
          Requires intervals long enough to hold several cycles.
    phase_alignment : "fixed" or "random", default="fixed"
        Whether the intervals are phase aligned across the dataset.

        * ``"fixed"`` -- all cases share the same interval positions (phase
          aligned). Interval-optimal: fixed-position regional features capture
          the signal.
        * ``"random"`` -- each case gets its own random interval positions
          (phase invariant), so the discriminative pattern can occur anywhere.
          This is the shapelet/convolution-optimal regime: fixed-position
          interval features cannot localise the signal, but phase-invariant
          methods (shapelets, convolution) can. Use it as the contrast to test
          whether a method's success depends on phase alignment.
    shapes : sequence of str or None, default=None
        Shape name per class, from ``SHAPES``. If None, the first
        ``n_classes`` distinct shapes are assigned. Must all differ (the classes
        are only separable if their shapes differ).
    amplitude : float, default=2.0
        Peak-to-base height of each shape.
    base : float, default=-1.0
        Baseline offset of each shape.
    noise_sigma : float, default=1.0
        Standard deviation of the additive i.i.d. Gaussian noise. Set to 0 for
        clean shapes (useful for sanity checks / visualisation).
    random_state : int, RandomState instance or None, default=None
        Controls interval placement, class-shape assignment and noise.
    return_params : bool, default=False
        If True, also return a dict describing the ground truth (interval start
        points, interval length, per-class shape names).

    Returns
    -------
    X : np.ndarray of shape (n_cases, 1, series_length)
        The simulated series (univariate).
    y : np.ndarray of shape (n_cases,)
        Integer class labels.
    params : dict, only if ``return_params``
        ``{"intervals": np.ndarray, "interval_length": int, "shapes": list}``.
    """
    rng = check_random_state(random_state)
    n_cases_per_class = list(n_cases_per_class)
    n_classes = len(n_cases_per_class)
    if n_classes < 2:
        raise ValueError("Need at least two classes.")

    if interval_length is None:
        interval_length = series_length // (n_intervals * noise_to_signal)
    if interval_length < 1:
        raise ValueError("Derived interval_length < 1; reduce noise_to_signal.")

    if discriminator not in ("shape", "scale", "trend", "frequency"):
        raise ValueError(
            f"Unknown discriminator '{discriminator}', "
            "valid: 'shape', 'scale', 'trend', 'frequency'."
        )

    if discriminator == "shape":
        if shapes is None:
            if n_classes > len(SHAPES):
                raise ValueError(
                    f"Only {len(SHAPES)} distinct shapes available for "
                    f"{n_classes} classes; pass shapes explicitly."
                )
            shapes = list(SHAPES)[:n_classes]
        else:
            shapes = list(shapes)
            if len(shapes) != n_classes:
                raise ValueError("shapes must have one entry per class.")
            if len(set(shapes)) < n_classes:
                raise ValueError("class shapes must be distinct to be separable.")
            for s in shapes:
                if s not in SHAPES:
                    raise ValueError(f"Unknown shape '{s}', valid: {tuple(SHAPES)}")
    else:
        shapes = None

    # per-class in-interval noise std (only differs by class for "scale")
    if discriminator == "scale":
        if interval_scales is None:
            # escalating regional variance, well separated: 2 sigma, 4 sigma, ...
            interval_scales = [noise_sigma * (2 + 2 * c) for c in range(n_classes)]
        interval_scales = list(interval_scales)
        if len(interval_scales) != n_classes:
            raise ValueError("interval_scales must have one entry per class.")

    # per-class sinusoid frequency (cycles per interval) for "frequency"
    if discriminator == "frequency":
        if frequencies is None:
            frequencies = [2 * (c + 1) for c in range(n_classes)]  # 2, 4, 6, ...
        frequencies = list(frequencies)
        if len(frequencies) != n_classes:
            raise ValueError("frequencies must have one entry per class.")
        if len(set(frequencies)) < n_classes:
            raise ValueError("class frequencies must be distinct to be separable.")
        if max(frequencies) >= interval_length / 2:
            raise ValueError(
                f"max frequency {max(frequencies)} cycles must be < "
                f"interval_length/2 ({interval_length / 2}) to be resolvable; "
                "use longer intervals."
            )

    if phase_alignment not in ("fixed", "random"):
        raise ValueError(
            f"Unknown phase_alignment '{phase_alignment}', "
            "valid: 'fixed', 'random'."
        )
    L = interval_length

    # per-class deterministic signal (shape/trend); None for stochastic modes
    signal_by_class = []
    for c in range(n_classes):
        if discriminator == "shape":
            signal_by_class.append(SHAPES[shapes[c]](L, amplitude, base))
        elif discriminator == "trend":
            slope = c - (n_classes - 1) / 2.0  # symmetric, zero-mean ramp
            signal_by_class.append(
                amplitude * slope * (np.arange(L) / max(L - 1, 1) - 0.5)
            )
        else:  # scale / frequency carry no fixed template
            signal_by_class.append(None)

    def _interval_signal(c):
        """Interval signal for class ``c`` (fresh per call for stochastic modes)."""
        if discriminator == "frequency":
            # sinusoid at the class frequency with random phase, so only the
            # spectral content is discriminative (not a matchable waveform)
            phase = rng.uniform(0, 2 * np.pi)
            return amplitude * np.sin(
                2 * np.pi * frequencies[c] * np.arange(L) / L + phase
            )
        return signal_by_class[c]

    def _stamp(starts, c):
        """Return (template, per-point sigma) for class ``c`` at ``starts``."""
        template = np.zeros(series_length, dtype=float)
        sig = np.full(series_length, noise_sigma, dtype=float)
        for s in starts:
            if discriminator == "scale":
                sig[s : s + L] = interval_scales[c]  # variance burst, no shape
            else:
                template[s : s + L] = _interval_signal(c)
        return template, sig

    # In "fixed" alignment the intervals are shared by every case (phase aligned
    # across the dataset -> interval-optimal). In "random" alignment each case
    # gets its own random interval positions (phase invariant -> the signal can
    # occur anywhere, so fixed-position interval features cannot localise it;
    # this is the shapelet/convolution-optimal regime).
    shared_starts = (
        _place_intervals(series_length, n_intervals, L, rng)
        if phase_alignment == "fixed"
        else None
    )

    # a deterministic template shared by all cases of a class is only valid when
    # positions are fixed AND the signal itself does not vary per case; "random"
    # phase and "frequency" (random sine phase) both need per-case generation.
    per_case = phase_alignment == "random" or discriminator == "frequency"

    X_list, y_list = [], []
    for c, n in enumerate(n_cases_per_class):
        cases = np.empty((n, series_length), dtype=float)
        if not per_case:
            template, sig = _stamp(shared_starts, c)
            cases[:] = template + rng.normal(0.0, 1.0, size=(n, series_length)) * sig
        else:
            for i in range(n):
                starts_i = (
                    shared_starts
                    if phase_alignment == "fixed"
                    else _place_intervals(series_length, n_intervals, L, rng)
                )
                template, sig = _stamp(starts_i, c)
                cases[i] = template + rng.normal(0.0, 1.0, size=series_length) * sig
        X_list.append(cases)
        y_list.append(np.full(n, c, dtype=int))

    X = np.vstack(X_list)[:, np.newaxis, :]  # (n_cases, 1, series_length)
    y = np.concatenate(y_list)

    # shuffle so classes are not blocked
    order = rng.permutation(len(y))
    X, y = X[order], y[order]

    if return_params:
        return X, y, {
            "intervals": shared_starts,  # None when phase_alignment="random"
            "interval_length": interval_length,
            "discriminator": discriminator,
            "phase_alignment": phase_alignment,
            "shapes": shapes,
            "interval_scales": interval_scales if discriminator == "scale" else None,
            "frequencies": frequencies if discriminator == "frequency" else None,
        }
    return X, y
