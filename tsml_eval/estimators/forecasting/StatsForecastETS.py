"""Wrapper for the statsforecast (Nixtla) AutoETS forecaster."""

import numpy as np
from aeon.forecasting import BaseForecaster


class StatsForecastETS(BaseForecaster):
    """aeon forecaster wrapping statsforecast's AutoETS.

    statsforecast's ``ETS`` is a deprecated alias of ``AutoETS``, so this is the
    statsforecast version of ETS. The model is fit once on the training series.

    ``predict(y)`` where ``y`` extends the training series by ``k`` values returns
    the fitted model's ``k + 1`` step ahead forecast, so the recursive fixed-horizon
    (M4) loop in ``run_forecasting_experiment``, which appends each prediction to the
    history, gets the model's genuine multi-step forecasts. Any other ``y`` is
    treated as a new series and the fitted model is applied to it via
    statsforecast's ``forward`` to get a one step ahead forecast.

    Parameters
    ----------
    season_length : int, default=1
        Number of observations per seasonal cycle. 1 means non-seasonal.
    model : str, default="ZZZ"
        Three letter statsforecast ETS model string (error, trend, season), each of
        "A" (additive), "M" (multiplicative), "N" (none) or "Z" (automatic).
    damped : bool or None, default=None
        Whether the trend is damped. None selects automatically.
    """

    _tags = {
        "capability:horizon": False,  # cannot fit to a horizon other than 1
    }

    def __init__(self, season_length=1, model="ZZZ", damped=None):
        self.season_length = season_length
        self.model = model
        self.damped = damped
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        from statsforecast.models import AutoETS

        self._y_fit = np.asarray(y, dtype=np.float64).squeeze(axis=0)
        self.model_ = AutoETS(
            season_length=int(self.season_length),
            model=self.model,
            damped=self.damped,
        )
        self.model_.fit(self._y_fit)
        self._forecasts = self.model_.predict(h=1)["mean"]
        self.forecast_ = float(self._forecasts[0])
        return self

    def _predict(self, y, exog=None):
        y = np.asarray(y, dtype=np.float64).squeeze(axis=0)
        n = len(self._y_fit)
        if len(y) >= n and np.array_equal(y[:n], self._y_fit):
            h = len(y) - n + 1
            if h > len(self._forecasts):
                self._forecasts = self.model_.predict(h=h)["mean"]
            return float(self._forecasts[h - 1])
        return float(self.model_.forward(y=y, h=1)["mean"][0])

    def _forecast(self, y, exog=None):
        self._fit(y, exog)
        return self.forecast_
