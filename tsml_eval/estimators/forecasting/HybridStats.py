from sklearn.ensemble import RandomForestRegressor
from aeon.forecasting import BaseForecaster, RegressionForecaster
from aeon.forecasting.stats import AutoETS, AutoARIMA
import numpy as np

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
    
class AverageStatsAIC(BaseForecaster):
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
        # self.auto_tar_model_ = AutoTAR()
        # self.auto_tar_model_.fit(y, exog=exog)
        return self

    def _predict(self, y, exog=None):
        ets_pred = self.ets_model_.predict(y, exog=exog)
        arima_pred = self.arima_model_.predict(y, exog=exog)
        # auto_tar_pred = self.auto_tar_model_.predict(y, exog=exog)
        return self._combine_forecasts(ets_pred, arima_pred)

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self._fit(y, exog)
        return self._combine_forecasts(self.ets_model_.forecast_, self.arima_model_.forecast_)

    def _combine_forecasts(self, ets_forecast, arima_forecast):
        """Combine the forecasts from the ETS, ARIMA, and AutoTAR models."""
        aics = np.array([
            self.ets_model_.wrapped_model_.aic_,
            self.arima_model_.final_model_.aic_,
            # self.auto_tar_model_.params_["selection"]["value"]
        ])

        forecasts = np.array([
            ets_forecast,
            arima_forecast,
            # auto_tar_forecast
        ])

        # convert AIC → weights (lower AIC = higher weight)
        weights = np.exp(-(aics - np.min(aics)) / 20)
        weights /= weights.sum()
        print(weights)
        print(forecasts)

        return np.dot(weights, forecasts)

class Ensemble1(BaseForecaster):
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
        self.random_forest_model_ = RegressionForecaster(window=100, regressor=RandomForestRegressor())
        self.random_forest_model_.fit(y, exog=exog)
        return self

    def _predict(self, y, exog=None):
        ets_pred = self.ets_model_.predict(y, exog=exog)
        arima_pred = self.arima_model_.predict(y, exog=exog)
        rf_pred = self.random_forest_model_.predict(y, exog=exog)
        return self._combine_forecasts(ets_pred, arima_pred, rf_pred)

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self._fit(y, exog)
        return self._combine_forecasts(self.ets_model_.forecast_, self.arima_model_.forecast_, self.random_forest_model_.forecast_)

    def _combine_forecasts(self, ets_forecast, arima_forecast, rf_forecast):
        """Combine the forecasts from the ETS, ARIMA, and Random Forest models."""
        return (ets_forecast + arima_forecast + rf_forecast) / 3

class EnsembleAIC1(BaseForecaster):
    """Test Hybrid Forecaster with alternate combination methods based on AIC."""

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
        self.random_forest_model_ = RegressionForecaster(window=100, regressor=RandomForestRegressor())
        self.random_forest_model_.fit(y, exog=exog)
        return self

    def _predict(self, y, exog=None):
        ets_pred = self.ets_model_.predict(y, exog=exog)
        arima_pred = self.arima_model_.predict(y, exog=exog)
        rf_pred = self.random_forest_model_.predict(y, exog=exog)
        return self._combine_forecasts(ets_pred, arima_pred, rf_pred)

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self._fit(y, exog)
        return self._combine_forecasts(self.ets_model_.forecast_, self.arima_model_.forecast_, self.random_forest_model_.forecast_)

    def _combine_forecasts(self, ets_forecast, arima_forecast, rf_forecast):
        """Combine the forecasts from the ETS, ARIMA, and Random Forest models."""
        return (ets_forecast + arima_forecast + rf_forecast) / 3