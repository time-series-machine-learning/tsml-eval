"""Guard against runaway iterative (recursive) forecasts from ML forecasters.

An iterative forecast feeds each one-step prediction back in as the next
observation. A machine learning one-step model can turn that feedback into an
exponential explosion, e.g. ``ESNormalisedRegressionForecaster`` on a series with a
one-off spike or a very noisy level: a spike-like prediction re-normalises the next
window and on some M4 series the forecast grows ~3x per step, to 1e10. Clipping the
regressor's output to its training targets cannot stop this, because the runaway
values are inside that range.

The guard keeps each step's prediction, before it is fed back, within a factor of
the training series' range, extended along its trend:

    upper_i = max(y) * factor * exp(max(g, 0) * i)
    lower_i = min(y) / factor * exp(min(g, 0) * i)

where ``g`` is the median one-step log growth of the training series and ``i`` the
step ahead (for a series that is not strictly positive the same band is built
additively: ``max(y) + (factor - 1) * range + max(g, 0) * i`` etc., ``g`` the median
one-step change). It is a sanity bound, not a forecasting device: a runaway forecast
is stopped at a few times the series' observed level, while a forecast inside the
band is exactly what the unguarded iterative loop would produce.

``factor=2`` was chosen on the M4 validation split (the last horizon of each
training series held out, so no test data): it improved the validation OWA of every
iterative ES-normalised Ridge and random forest configuration tried, slightly more
than ``factor=3`` (and the unguarded loop was worst). The actual future values of
98.9% of M4 series (99.6% for ``factor=3``) lie inside the ``factor=2`` band.
"""

__all__ = [
    "ITERATIVE_GUARD_FACTOR",
    "iterative_guard_bounds",
    "guarded_iterative_forecast",
    "is_ml_forecaster",
]

import numpy as np

ITERATIVE_GUARD_FACTOR = 2.0


def iterative_guard_bounds(y, prediction_horizon, factor=ITERATIVE_GUARD_FACTOR):
    """Return ``(lower, upper)`` arrays bounding each step of an iterative forecast.

    Parameters
    ----------
    y : array-like
        The training series the forecast starts from.
    prediction_horizon : int
        Number of steps forecast.
    factor : float, default=ITERATIVE_GUARD_FACTOR
        How far beyond the training range (multiplicatively for positive series)
        a forecast may go, on top of the trend.

    Returns
    -------
    lower, upper : np.ndarray of shape (prediction_horizon,)
    """
    y = np.asarray(y, dtype=float).ravel()
    steps = np.arange(1, prediction_horizon + 1)
    lo, hi = float(np.min(y)), float(np.max(y))
    if lo > 0:
        g = float(np.median(np.diff(np.log(y)))) if len(y) > 1 else 0.0
        lower = lo / factor * np.exp(min(g, 0.0) * steps)
        upper = hi * factor * np.exp(max(g, 0.0) * steps)
    else:
        g = float(np.median(np.diff(y))) if len(y) > 1 else 0.0
        margin = (factor - 1) * max(hi - lo, 1e-8)
        lower = lo - margin + min(g, 0.0) * steps
        upper = hi + margin + max(g, 0.0) * steps
    return lower, upper


def guarded_iterative_forecast(
    forecaster, y, prediction_horizon, factor=ITERATIVE_GUARD_FACTOR
):
    """Iterative forecast with each step bounded by ``iterative_guard_bounds``.

    The same loop as aeon's ``IterativeForecastingMixin.iterative_forecast`` (fit
    once on ``y``, then predict one step and append it to the history,
    ``prediction_horizon`` times), except each prediction is clipped to the guard
    band before it is appended. Steps inside the band are unchanged, so a forecast
    the guard never touches is identical to the unguarded one.

    Parameters
    ----------
    forecaster : aeon one-step-ahead forecaster (``horizon=1``)
    y : np.ndarray
        Training series.
    prediction_horizon : int
    factor : float, default=ITERATIVE_GUARD_FACTOR

    Returns
    -------
    preds : np.ndarray of shape (prediction_horizon,)
    n_clipped : int
        Number of steps the guard clipped.
    """
    if getattr(forecaster, "horizon", 1) != 1:
        raise ValueError(
            "guarded_iterative_forecast requires a one-step-ahead forecaster "
            f"(horizon=1), got horizon={forecaster.horizon}."
        )
    y, _, _ = forecaster._check_iterative_forecast_inputs(y, prediction_horizon)
    lower, upper = iterative_guard_bounds(y, prediction_horizon, factor)
    preds = np.zeros(prediction_horizon)
    n_clipped = 0
    forecaster.fit(y)
    for i in range(prediction_horizon):
        p = float(forecaster.predict(y))
        c = float(np.clip(p, lower[i], upper[i]))
        n_clipped += c != p
        preds[i] = c
        y = np.append(y, c)
    return preds, int(n_clipped)


def is_ml_forecaster(forecaster):
    """Whether ``forecaster`` is a machine learning (regression-based) forecaster.

    True for forecasters built on a regressor (``RegressionForecaster``,
    ``ESNormalisedRegressionForecaster``, wrappers of them such as
    ``AdaptiveWindowEnsemble``) and for aeon's machine/deep learning forecasters.
    Statistical models (ETS, ARIMA, Theta, ...) are not guarded.
    """
    if hasattr(forecaster, "regressor"):
        return True
    inner = getattr(forecaster, "forecaster", None)
    if inner is not None and is_ml_forecaster(inner):
        return True
    module = type(forecaster).__module__
    return ".machine_learning." in module or ".deep_learning." in module
