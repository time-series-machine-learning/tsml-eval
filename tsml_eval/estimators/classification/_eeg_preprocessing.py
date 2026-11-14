"""Preprocessing for EEG classifiers: band-pass filtering and normalisation.

``get_classifier_by_name`` puts these in front of any classifier by name suffix:
``-BP830`` (8-30 Hz), ``-BP140`` (1-40 Hz) and ``-BP640`` (6-40 Hz) band-pass each channel and need the
dataset's sampling rate as ``sfreq``; ``-norm`` z-normalises each channel of each
case. For example ``RiemannianMDM-OAS-BP830`` or ``CNN-norm``.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["BandPassFilter", "BANDS", "preprocessed_classifier"]

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer

#: name suffix -> (low, high) cut-offs in Hz
# bp640 (6-40 Hz) is for steady-state visual evoked potentials, e.g. TRCA-BP640
BANDS = {"bp830": (8.0, 30.0), "bp140": (1.0, 40.0), "bp640": (6.0, 40.0)}


class BandPassFilter(BaseCollectionTransformer):
    """Zero-phase Butterworth band-pass filter, applied to every channel.

    A forward-backward (``scipy.signal.sosfiltfilt``) filter, so the output has no
    phase shift and the same length as the input. Each case is filtered on its
    own, so nothing is learnt from the data and train and test are treated alike.
    Short series are padded with ``min(default, n_timepoints - 1)`` points.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data in Hz.
    l_freq : float, default=8.0
        Lower cut-off in Hz, above 0.
    h_freq : float, default=30.0
        Upper cut-off in Hz, below ``sfreq / 2``.
    order : int, default=4
        Butterworth order of each pass (the forward-backward filter doubles it).
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "fit_is_empty": True,
        "capability:multivariate": True,
    }

    def __init__(self, sfreq, l_freq=8.0, h_freq=30.0, order=4):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.order = order
        super().__init__()

    def _sos(self):
        from scipy.signal import butter

        sfreq, low, high = float(self.sfreq), float(self.l_freq), float(self.h_freq)
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError(f"sfreq must be a positive number of Hz, not {sfreq}")
        if not 0 < low < high < sfreq / 2:
            raise ValueError(
                f"Need 0 < l_freq < h_freq < sfreq / 2, got {low}, {high} with "
                f"sfreq {sfreq}"
            )
        return butter(self.order, [low, high], btype="band", fs=sfreq, output="sos")

    def _transform(self, X, y=None):
        from scipy.signal import sosfiltfilt

        sos = self._sos()
        default = 3 * (
            2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
        )
        padlen = min(default, X.shape[-1] - 1)
        return sosfiltfilt(sos, np.asarray(X, dtype=np.float64), axis=-1, padlen=padlen)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"sfreq": 100.0, "l_freq": 8.0, "h_freq": 30.0}


def preprocessed_classifier(name, build, random_state, kwargs):
    """Return a pipeline for a suffixed name, or None without a known suffix.

    Parameters
    ----------
    name : str
        The lower-case classifier name, e.g. ``riemannianmdm-oas-bp830``.
    build : callable
        ``build(base_name, kwargs)`` returns the classifier without the suffix.
    random_state : int or None
        Passed to the pipeline.
    kwargs : dict
        Keyword arguments for the classifier; ``sfreq`` is taken for the filter.
    """
    base, _, suffix = name.rpartition("-")
    if not base or (suffix not in BANDS and suffix != "norm"):
        return None
    kwargs = dict(kwargs)
    if suffix in BANDS:
        if "sfreq" not in kwargs:
            raise ValueError(
                f"{name} band-pass filters the data, so it needs the sampling rate: "
                "pass sfreq=<Hz>"
            )
        low, high = BANDS[suffix]
        transformer = BandPassFilter(float(kwargs.pop("sfreq")), low, high)
    else:
        from aeon.transformations.collection import Normalizer

        transformer = Normalizer()
    from aeon.classification.compose import ClassifierPipeline

    return ClassifierPipeline(
        [transformer], build(base, kwargs), random_state=random_state
    )
