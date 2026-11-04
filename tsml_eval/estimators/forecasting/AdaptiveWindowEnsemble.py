"""Average a windowed forecaster over several adaptive window rules."""

__all__ = ["AdaptiveWindowEnsemble"]

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin

from tsml_eval.estimators.forecasting._adaptive_window import adaptive_window_length
from tsml_eval.estimators.forecasting._iterative_guard import (
    guarded_iterative_forecast,
)


class AdaptiveWindowEnsemble(BaseForecaster, IterativeForecastingMixin):
    """Mean forecast of copies of a forecaster using different window lengths.

    Each member is a clone of ``forecaster`` whose ``window`` is set, per series, by
    one of ``window_rules`` (see ``adaptive_window_length``). Different windows make
    different errors, so averaging them is a cheap way to reduce variance, e.g. for
    ``ESNormalisedRegressionForecaster`` with a ridge regressor.

    Parameters
    ----------
    forecaster : BaseForecaster
        A one-step-ahead forecaster with ``window`` and (optionally)
        ``seasonal_period`` attributes. Its ``window`` is the maximum window.
    window_rules : tuple of str, default=("quarter", "seasonal")
        One member per rule.
    """

    _tags = {
        "capability:horizon": False,
    }

    def __init__(self, forecaster, window_rules=("quarter", "seasonal")):
        self.forecaster = forecaster
        self.window_rules = window_rules
        super().__init__(horizon=1, axis=1)

    def _members(self, fit_length, full_horizon):
        members = []
        for rule in self.window_rules:
            f = _clone_estimator(self.forecaster)
            f.window = adaptive_window_length(
                rule,
                self.forecaster.window,
                fit_length,
                step_horizon=getattr(f, "horizon", 1) or 1,
                full_horizon=full_horizon,
                seasonal_period=getattr(f, "seasonal_period", 1) or 1,
            )
            members.append(f)
        return members

    def _fit(self, y, exog=None):
        y = np.asarray(y).squeeze()
        self.forecasters_ = [f.fit(y) for f in self._members(len(y), 1)]
        self.forecast_ = float(np.mean([f.forecast_ for f in self.forecasters_]))
        return self

    def _predict(self, y, exog=None):
        y = np.asarray(y).squeeze()
        return float(np.mean([f.predict(y) for f in self.forecasters_]))

    def iterative_forecast(self, y, prediction_horizon, exog=None, *, future_exog=None):
        """Mean of each member's iterative forecast, windows set for this horizon."""
        y = np.asarray(y, dtype=float).squeeze()
        self.forecasters_ = self._members(len(y), prediction_horizon)
        # Each member's recursion is bounded by the ML iterative guard (see
        # _iterative_guard). A member can still fail on a series (e.g. a non-finite
        # prediction). Average the members that succeed, and only fail if none do.
        preds, errors = [], []
        self.n_guard_clipped_ = 0
        for f in self.forecasters_:
            try:
                p, n_clipped = guarded_iterative_forecast(f, y, prediction_horizon)
            except ValueError as e:
                errors.append(e)
                continue
            self.n_guard_clipped_ += n_clipped
            if np.isfinite(p).all():
                preds.append(p)
        if not preds:
            raise ValueError(
                f"All {len(self.forecasters_)} ensemble members failed: {errors}"
            )
        self.n_members_used_ = len(preds)
        return np.mean(preds, axis=0)
