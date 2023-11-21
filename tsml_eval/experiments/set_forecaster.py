"""Set forecaster function."""

__author__ = ["MatthewMiddlehurst"]

from aeon.forecasting.naive import NaiveForecaster

from tsml_eval.utils.functions import str_in_nested_list

ml_forecasters = [
    ["lr", "linearregression"],
    "1nn",
    ["rf", "randomforest"],
    "xgboost",
    "inceptiontime",
    "rocket",
    "freshprince",
    "drcif",
    ["rotf", "rotationforest"],
]
other_forecasters = [
    ["naiveforecaster", "naive"],
]


def set_forecaster(forecaster_name, random_state=None, n_jobs=1, **kwargs):
    """Return a forecaster matching a given input name.

    Basic way of creating a forecaster to build using the default or alternative
    settings. This set up is to help with batch jobs for multiple problems and to
    facilitate easy reproducibility for use with run_forecasting_experiment.

    Generally, inputting a forecasters class name will return said forecaster with
    default settings.

    todo

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

    if str_in_nested_list(ml_forecasters, f):
        return _set_forecaster_ml(f, random_state, n_jobs, kwargs)
    elif str_in_nested_list(other_forecasters, f):
        return _set_forecaster_other(f, random_state, n_jobs, kwargs)
    else:
        raise ValueError(f"UNKNOWN FORECASTER: {f} in set_forecaster")


def _set_forecaster_ml(f, random_state, n_jobs, kwargs):
    if f == "lr" or f == "linearregression":
        from aeon.forecasting.compose import make_reduction
        from aeon.forecasting.trend import PolynomialTrendForecaster
        from aeon.pipeline import make_pipeline
        from aeon.transformations.series.detrend import Detrender
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        regressor = LinearRegression(n_jobs=n_jobs, **kwargs)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "1nn":
        from aeon.forecasting.compose import make_reduction
        from aeon.forecasting.trend import PolynomialTrendForecaster
        from aeon.pipeline import make_pipeline
        from aeon.transformations.series.detrend import Detrender
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler

        regressor = KNeighborsRegressor(n_neighbors=1, n_jobs=n_jobs, **kwargs)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "rf" or f == "randomforest":
        from aeon.forecasting.compose import make_reduction
        from aeon.forecasting.trend import PolynomialTrendForecaster
        from aeon.pipeline import make_pipeline
        from aeon.transformations.series.detrend import Detrender
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        regressor = RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "xgboost":
        from aeon.forecasting.compose import make_reduction
        from aeon.forecasting.trend import PolynomialTrendForecaster
        from aeon.pipeline import make_pipeline
        from aeon.transformations.series.detrend import Detrender
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor

        regressor = XGBRegressor(
            n_estimators=200, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )

    elif f == "inceptiontime":
        from aeon.forecasting.compose import make_reduction
        from aeon.forecasting.trend import PolynomialTrendForecaster
        from aeon.pipeline import make_pipeline
        from aeon.regression.deep_learning import InceptionTimeRegressor
        from aeon.transformations.series.detrend import Detrender
        from sklearn.preprocessing import StandardScaler

        regressor = InceptionTimeRegressor(random_state=random_state, **kwargs)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "rocket":
        from aeon.forecasting.compose import make_reduction
        from aeon.forecasting.trend import PolynomialTrendForecaster
        from aeon.pipeline import make_pipeline
        from aeon.regression.convolution_based import RocketRegressor
        from aeon.transformations.series.detrend import Detrender
        from sklearn.preprocessing import StandardScaler

        regressor = RocketRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "freshprince":
        from aeon.forecasting.compose import make_reduction
        from aeon.forecasting.trend import PolynomialTrendForecaster
        from aeon.pipeline import make_pipeline
        from aeon.regression.feature_based import FreshPRINCERegressor
        from aeon.transformations.series.detrend import Detrender
        from sklearn.preprocessing import StandardScaler

        regressor = FreshPRINCERegressor(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "drcif":
        from aeon.forecasting.compose import make_reduction
        from aeon.forecasting.trend import PolynomialTrendForecaster
        from aeon.pipeline import make_pipeline
        from aeon.regression.interval_based import DrCIFRegressor
        from aeon.transformations.series.detrend import Detrender
        from sklearn.preprocessing import StandardScaler

        regressor = DrCIFRegressor(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )
    elif f == "rotf" or f == "rotationforest":
        from aeon.forecasting.compose import make_reduction
        from aeon.forecasting.trend import PolynomialTrendForecaster
        from aeon.pipeline import make_pipeline
        from aeon.regression.sklearn import RotationForestRegressor
        from aeon.transformations.series.detrend import Detrender
        from sklearn.preprocessing import StandardScaler

        regressor = RotationForestRegressor(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
        return make_pipeline(
            Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
            StandardScaler(),
            make_reduction(regressor, window_length=15, strategy="recursive"),
        )


def _set_forecaster_other(f, random_state, n_jobs, kwargs):
    if f == "naiveforecaster" or f == "naive":
        return NaiveForecaster(**kwargs)
