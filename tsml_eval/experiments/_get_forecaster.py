"""Set forecaster function."""

__maintainer__ = ["MatthewMiddlehurst"]

from tsml_eval.experiments._get_regressor import get_regressor_by_name
from tsml_eval.utils.functions import str_in_nested_list

deep_forecasters = [
    ["tcnforecaster", "tcn"],
]
ml_forecasters = [
    "setartree",
    "setarforest",
]
stats_forecasters = [
    ["arimaforecaster", "arima"],
    "autoarima",
    ["etsforecaster", "ets"],
    ["tarforecaster", "tar"],
    "autotar",
    ["setarforecaster", "setar"],
    ["thetaforecaster", "theta"],
    ["tvpforecaster", "tvp"],
    ["averagestats", "average", "hybridaverage"],
]
regression_forecasters = [
    "randomforest",
]
other_forecasters = [
    ["naiveforecaster", "naive"],
]


def get_forecaster_by_name(forecaster_name, random_state=None, n_jobs=1, **kwargs):
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

    if str_in_nested_list(deep_forecasters, f):
        return _set_forecaster_deep(f, random_state, n_jobs, kwargs)
    elif str_in_nested_list(ml_forecasters, f):
        return _set_forecaster_ml(f, random_state, n_jobs, kwargs)
    elif str_in_nested_list(stats_forecasters, f):
        return _set_forecaster_stats(f, random_state, n_jobs, kwargs)
    elif str_in_nested_list(regression_forecasters, f):
        return _set_forecaster_regression(f, random_state, n_jobs, kwargs)
    elif str_in_nested_list(other_forecasters, f):
        return _set_forecaster_other(f, random_state, n_jobs, kwargs)
    else:
        window = 100
        print(kwargs)
        if 'window' in kwargs:
            window = kwargs.pop('window')
        try:
            regressor = get_regressor_by_name(f, random_state, n_jobs, **kwargs)
        except ValueError:
            raise ValueError(f"UNKNOWN FORECASTER: {f} in get_forecaster_by_name")
        from aeon.forecasting import RegressionForecaster
        return RegressionForecaster(window=window, regressor=regressor)


def _set_forecaster_deep(f, random_state, n_jobs, kwargs):
    if f == "tcnforecaster" or f == "tcn":
        from aeon.forecasting.deep_learning import TCNForecaster

        return TCNForecaster(random_state=random_state, **kwargs)


def _set_forecaster_ml(f, random_state, n_jobs, kwargs):
    if f == "setartree":
        from aeon.forecasting.machine_learning import SETARTree

        return SETARTree(**kwargs)
    elif f == "setarforest":
        from aeon.forecasting.machine_learning import SETARForest

        return SETARForest(random_state=random_state, **kwargs)


def _set_forecaster_stats(f, random_state, n_jobs, kwargs):
    if f == "arimaforecaster" or f == "arima":
        from aeon.forecasting.stats import ARIMA

        return ARIMA(**kwargs)
    elif f == "autoarima":
        from aeon.forecasting.stats import AutoARIMA

        return AutoARIMA(**kwargs)
    elif f == "etsforecaster" or f == "ets":
        from aeon.forecasting.stats import ETS

        return ETS(**kwargs)
    elif f == "tarforecaster" or f == "tar":
        from aeon.forecasting.stats import TAR

        return TAR(**kwargs)
    elif f == "autotar":
        from aeon.forecasting.stats import AutoTAR

        return AutoTAR(**kwargs)
    elif f == "setarforecaster" or f == "setar":
        from aeon.forecasting.machine_learning import SETAR

        return SETAR(**kwargs)
    elif f == "thetaforecaster" or f == "theta":
        from aeon.forecasting.stats import Theta

        return Theta(**kwargs)
    elif f == "tvpforecaster" or f == "tvp":
        from aeon.forecasting.stats import TVP

        return TVP(**kwargs)
    elif f == "averagestats" or f == "average" or f == "hybridaverage":
        from tsml_eval.estimators.forecasting.HybridStats import AverageStats

        return AverageStats(**kwargs)


def _set_forecaster_regression(f, random_state, n_jobs, kwargs):
    if f == "randomforest":
        from aeon.forecasting import RegressionForecaster
        from sklearn.ensemble import RandomForestRegressor

        reg = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
        return RegressionForecaster(10, regressor=reg)


def _set_forecaster_other(f, random_state, n_jobs, kwargs):
    if f == "naiveforecaster" or f == "naive":
        from aeon.forecasting import NaiveForecaster

        return NaiveForecaster(**kwargs)
