"""Rules for shrinking a forecaster's window to suit the length of a series."""

__all__ = ["ADAPTIVE_WINDOW_RULES", "adaptive_window_length"]

ADAPTIVE_WINDOW_RULES = ("half", "quarter", "seasonal")


def adaptive_window_length(
    rule,
    window,
    fit_length,
    step_horizon=1,
    full_horizon=None,
    seasonal_period=1,
    min_window=1,
):
    """Return the window to use for a series with ``fit_length`` training points.

    Parameters
    ----------
    rule : bool or str
        ``True`` or ``"half"``: the original rule, the window is halved to
        ``fit_length // 2`` when the series is shorter than twice the window.
        ``"quarter"``: at most a quarter of the points left once the full forecast
        horizon is removed, ``(fit_length - full_horizon) // 4``. This leaves many more
        training windows than ``"half"`` on short series, so a model is not fit on a
        handful of samples.
        ``"seasonal"``: about ``1.25 * max(seasonal_period, full_horizon)`` (the
        ES-RNN input size), at most half the points left after the full horizon.
        In every case the result is at most ``window`` and at least 1.
    window : int
        The configured (maximum) window.
    fit_length : int
        Length of the series the forecaster is fit on.
    step_horizon : int, default=1
        Horizon of the forecaster itself (1 for one-step-ahead models).
    full_horizon : int or None, default=None
        Number of steps that will be forecast (e.g. the M4 horizon). Defaults to
        ``step_horizon``.
    seasonal_period : int, default=1
        Seasonal period of the series, used by ``"seasonal"``.
    min_window : int, default=1
        Lower bound applied after the rule (but never above ``fit_length - 3``),
        e.g. 5 for regressors using catch22 features, which need 5 points.
    """
    w = _rule_window(rule, window, fit_length, step_horizon, full_horizon,
                     seasonal_period)
    if min_window > 1:
        w = max(w, min(min_window, fit_length - 3))
    return w


def _rule_window(rule, window, fit_length, step_horizon, full_horizon,
                 seasonal_period):
    if full_horizon is None:
        full_horizon = step_horizon
    # Largest window that keeps at least 3 training windows (as RegressionForecaster)
    max_window = fit_length - step_horizon - 3
    if rule is True or rule == "half":
        if fit_length < 2 * window or window > max_window:
            return max(1, min(fit_length // 2, max_window))
        return window
    remaining = fit_length - full_horizon
    if rule == "quarter":
        w = remaining // 4
    elif rule == "seasonal":
        w = min(int(round(1.25 * max(seasonal_period, full_horizon))), remaining // 2)
    else:
        raise ValueError(
            f"Unknown adaptive window rule {rule!r}, must be True or one of "
            f"{ADAPTIVE_WINDOW_RULES}."
        )
    return max(1, min(window, w, max(1, max_window)))
