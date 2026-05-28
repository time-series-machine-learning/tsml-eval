from sklearn.ensemble import RandomForestRegressor
from aeon.forecasting import BaseForecaster, RegressionForecaster
from aeon.forecasting.stats import AutoETS, AutoARIMA, AutoTAR, Theta, TVP
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from aeon.regression.sklearn import RotationForestRegressor
from aeon.regression.interval_based import TimeSeriesForestRegressor
from aeon.regression.deep_learning import InceptionTimeRegressor
from aeon.regression.deep_learning import ResNetRegressor
from aeon.regression.deep_learning import TimeCNNRegressor
from aeon.regression.deep_learning import LITETimeRegressor
from aeon.regression.deep_learning import DisjointCNNRegressor
from aeon.regression.convolution_based import MultiRocketRegressor
from aeon.regression.convolution_based import HydraRegressor
from aeon.regression.feature_based import FreshPRINCERegressor

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
        weights = np.exp(-(aics - np.min(aics)) / 500)
        weights /= weights.sum()
        print(weights)
        print(forecasts)

        return np.dot(weights, forecasts)

def median(forecasts):
    """Compute the median of the forecasts."""
    return np.median(forecasts)

def mean(forecasts):
    """Compute the mean of the forecasts."""
    return np.mean(forecasts)

def middle(forecasts):
    """Compute the middle of the forecasts."""
    forecasts.sort()
    return np.median(forecasts) if len(forecasts) % 2 == 1 else (sum(forecasts[len(forecasts) // 2 - 1:len(forecasts) // 2 + 2]) / 3)

class Ensemble1(BaseForecaster):
    """Test Hybrid Forecaster."""

    _tags = {
        "capability:horizon": False,  # cannot fit to a horizon other than 1
    }

    def __init__(
        self,
    ):
        super().__init__(horizon=1, axis=1)
        self.combination_method = median

    def _fit(self, y, exog=None):
        self.models = []
        # self.models.append(AutoETS())
        # self.models.append(AutoARIMA())
        # self.models.append(AutoTAR(None, None, None))
        # self.models.append(Theta())
        # self.models.append(TVP(window=100))
        self.models.append(RegressionForecaster(window=100, regressor=RandomForestRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=RidgeCV(fit_intercept=True, alphas=np.logspace(-3, 3, 10))))
        self.models.append(RegressionForecaster(window=100, regressor=XGBRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=RotationForestRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=TimeSeriesForestRegressor()))
        for model in self.models:
            model.fit(y, exog=exog)
        return self

    def _predict(self, y, exog=None):
        return self._combine_forecasts([model.predict(y, exog=exog) for model in self.models])

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self._fit(y, exog)
        return self._combine_forecasts([model.forecast_ for model in self.models])

    def _combine_forecasts(self, forecasts):
        """Combine the forecasts from the models."""
        return self.combination_method(forecasts)

class Ensemble2(BaseForecaster):
    """Test Hybrid Forecaster."""

    _tags = {
        "capability:horizon": False,  # cannot fit to a horizon other than 1
    }

    def __init__(
        self,
    ):
        super().__init__(horizon=1, axis=1)
        self.combination_method = median

    def _fit(self, y, exog=None):
        self.models = []
        # self.models.append(AutoETS())
        # self.models.append(AutoARIMA())
        # self.models.append(RegressionForecaster(window=100, regressor=RandomForestRegressor()))
        # self.models.append(RegressionForecaster(window=100, regressor=RidgeCV(fit_intercept=True, alphas=np.logspace(-3, 3, 10))))
        # self.models.append(RegressionForecaster(window=100, regressor=XGBRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=MultiRocketRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=HydraRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=FreshPRINCERegressor()))
        for model in self.models:
            model.fit(y, exog=exog)
        return self

    def _predict(self, y, exog=None):
        return self._combine_forecasts([model.predict(y, exog=exog) for model in self.models])

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self._fit(y, exog)
        return self._combine_forecasts([model.forecast_ for model in self.models])

    def _combine_forecasts(self, forecasts):
        """Combine the forecasts from the models."""
        return self.combination_method(forecasts)

class Ensemble3(BaseForecaster):
    """Test Hybrid Forecaster."""

    _tags = {
        "capability:horizon": False,  # cannot fit to a horizon other than 1
    }

    def __init__(
        self,
    ):
        super().__init__(horizon=1, axis=1)
        self.combination_method = median

    def _fit(self, y, exog=None):
        self.models = []
        # self.models.append(AutoETS())
        # self.models.append(AutoARIMA())
        # self.models.append(AutoTAR(None, None, None))
        # self.models.append(Theta())
        # self.models.append(TVP(window=100))
        # self.models.append(RegressionForecaster(window=100, regressor=RandomForestRegressor()))
        # self.models.append(RegressionForecaster(window=100, regressor=RidgeCV(fit_intercept=True, alphas=np.logspace(-3, 3, 10))))
        # self.models.append(RegressionForecaster(window=100, regressor=XGBRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=InceptionTimeRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=ResNetRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=TimeCNNRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=LITETimeRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=DisjointCNNRegressor()))
        for model in self.models:
            model.fit(y, exog=exog)
        return self

    def _predict(self, y, exog=None):
        return self._combine_forecasts([model.predict(y, exog=exog) for model in self.models])

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self._fit(y, exog)
        return self._combine_forecasts([model.forecast_ for model in self.models])

    def _combine_forecasts(self, forecasts):
        """Combine the forecasts from the models."""
        return self.combination_method(forecasts)

class Ensemble4(BaseForecaster):
    """Test Hybrid Forecaster."""

    _tags = {
        "capability:horizon": False,  # cannot fit to a horizon other than 1
    }

    def __init__(
        self,
    ):
        super().__init__(horizon=1, axis=1)
        self.combination_method = middle

    def _fit(self, y, exog=None):
        self.models = []
        self.models.append(AutoETS())
        self.models.append(AutoARIMA())
        # self.models.append(RegressionForecaster(window=100, regressor=RandomForestRegressor()))
        self.models.append(RegressionForecaster(window=100, regressor=RidgeCV(fit_intercept=True, alphas=np.logspace(-3, 3, 10))))
        self.models.append(RegressionForecaster(window=100, regressor=XGBRegressor()))
        for model in self.models:
            model.fit(y, exog=exog)
        return self

    def _predict(self, y, exog=None):
        return self._combine_forecasts([model.predict(y, exog=exog) for model in self.models])

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self._fit(y, exog)
        return self._combine_forecasts([model.forecast_ for model in self.models])

    def _combine_forecasts(self, forecasts):
        """Combine the forecasts from the models."""
        return self.combination_method(forecasts)
    
class Ensemble5(BaseForecaster):
    """Test Hybrid Forecaster."""

    _tags = {
        "capability:horizon": False,  # cannot fit to a horizon other than 1
    }

    def __init__(
        self,
    ):
        super().__init__(horizon=1, axis=1)
        self.combination_method = middle

    def _fit(self, y, exog=None):
        self.models = []
        self.models.append(AutoETS())
        self.models.append(AutoARIMA())
        self.models.append(AutoTAR(None, None, None))
        self.models.append(Theta())
        self.models.append(TVP(window=100))
        # self.models.append(RegressionForecaster(window=100, regressor=RandomForestRegressor()))
        # self.models.append(RegressionForecaster(window=100, regressor=RidgeCV(fit_intercept=True, alphas=np.logspace(-3, 3, 10))))
        # self.models.append(RegressionForecaster(window=100, regressor=XGBRegressor()))
        for model in self.models:
            model.fit(y, exog=exog)
        return self

    def _predict(self, y, exog=None):
        return self._combine_forecasts([model.predict(y, exog=exog) for model in self.models])

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self._fit(y, exog)
        return self._combine_forecasts([model.forecast_ for model in self.models])

    def _combine_forecasts(self, forecasts):
        """Combine the forecasts from the models."""
        return self.combination_method(forecasts)

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