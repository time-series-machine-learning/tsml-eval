from aeon.forecasting import BaseForecaster
from aeon.forecasting.stats import AutoETS, AutoARIMA

class AverageStats(BaseForecaster):
    """Test Hybrid Forecaster."""

    _tags = {
        "capability:horizon": False,  # cannot fit to a horizon other than 1
    }

    def __init__(
        self,
    ):
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        self.ets_model_ = AutoETS()
        self.ets_model_.fit(y, exog=exog)
        self.arima_model_ = AutoARIMA()
        self.arima_model_.fit(y, exog=exog)
        return self

    def _predict(self, y, exog=None):
        ets_pred = self.ets_model_.predict(y, exog=exog)
        arima_pred = self.arima_model_.predict(y, exog=exog)
        return self._combine_forecasts(ets_pred, arima_pred)

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self._fit(y, exog)
        return self._combine_forecasts(self.ets_model_.forecast_, self.arima_model_.forecast_)

    def _combine_forecasts(self, ets_forecast, arima_forecast):
        """Combine the forecasts from the ETS and ARIMA models."""
        return (ets_forecast + arima_forecast) / 2