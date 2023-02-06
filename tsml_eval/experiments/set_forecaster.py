# -*- coding: utf-8 -*-
"""Set forecaster function."""

__author__ = ["MatthewMiddlehurst"]


def set_forecaster(forecaster_name, random_state=None, n_jobs=1):
    """Return a forecaster matching a given input name.

    Basic way of creating a forecaster to build using the default or alternative
    settings. This set up is to help with batch jobs for multiple problems and to
    facilitate easy reproducibility for use with run_classification_experiment.

    Generally, inputting a forecasters class name will return said classifier with
    default settings.

    Parameters
    ----------
    forecaster_name : str
        String indicating which classifier to be returned.
    random_state : int, RandomState instance or None, default=None
        Random seed or RandomState object to be used in the classifier if available.
    n_jobs: int, default=1
        The number of jobs to run in parallel for both classifier ``fit`` and
        ``predict`` if available. `-1` means using all processors.

    Return
    ------
    forecaster : A BaseForecaster.
        The forecaster matching the input classifier name.
    """
    f = forecaster_name.lower()

    if f == "lr" or f == "linearregression":
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.pipeline import make_pipeline
        from sktime.transformations.series.detrend import Detrender

        regressor = LinearRegression(n_jobs=n_jobs)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "1nn":
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.pipeline import make_pipeline
        from sktime.transformations.series.detrend import Detrender

        regressor = KNeighborsRegressor(n_neighbors=1, n_jobs=n_jobs)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "rf" or f == "randomforest":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.pipeline import make_pipeline
        from sktime.transformations.series.detrend import Detrender

        regressor = RandomForestRegressor(n_estimators=200, n_jobs=n_jobs)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "xgboost":
        from sklearn.preprocessing import StandardScaler
        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.pipeline import make_pipeline
        from sktime.transformations.series.detrend import Detrender
        from xgboost import XGBRegressor

        regressor = XGBRegressor(n_estimators=200, n_jobs=n_jobs)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )

    elif f == "inceptiontime":
        from sklearn.preprocessing import StandardScaler
        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.pipeline import make_pipeline
        from sktime.transformations.series.detrend import Detrender

        from tsml_eval.sktime_estimators.regression.deep_learning import (
            InceptionTimeRegressor,
        )

        regressor = InceptionTimeRegressor(random_state=random_state)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "rocket":
        from sklearn.preprocessing import StandardScaler
        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.pipeline import make_pipeline
        from sktime.regression.kernel_based import RocketRegressor
        from sktime.transformations.series.detrend import Detrender

        regressor = RocketRegressor(random_state=random_state, n_jobs=n_jobs)
        return make_reduction(regressor, window_length=15, strategy="recursive")
    elif f == "freshprince":
        from sklearn.preprocessing import StandardScaler
        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.pipeline import make_pipeline
        from sktime.transformations.series.detrend import Detrender

        from tsml_eval.sktime_estimators.regression.featured_based import (
            FreshPRINCERegressor,
        )

        regressor = FreshPRINCERegressor(random_state=random_state, n_jobs=n_jobs)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "drcif":
        from sklearn.preprocessing import StandardScaler
        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.pipeline import make_pipeline
        from sktime.transformations.series.detrend import Detrender

        from tsml_eval.sktime_estimators.regression.interval_based import DrCIF

        regressor = DrCIF(n_estimators=500, random_state=random_state, n_jobs=n_jobs)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "rotf" or f == "rotationforest":
        from sklearn.preprocessing import StandardScaler
        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.pipeline import make_pipeline
        from sktime.transformations.series.detrend import Detrender

        from tsml_eval.sktime_estimators.regression.sklearn import RotationForest

        regressor = RotationForest(random_state=random_state, n_jobs=n_jobs)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )

    # invalid regressor
    else:
        raise Exception("UNKNOWN REGRESSOR ", f, " in set_regressor")
