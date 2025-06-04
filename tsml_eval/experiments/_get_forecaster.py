"""Set forecaster function."""

__maintainer__ = ["MatthewMiddlehurst"]

from aeon.forecasting import (
    AutoSARIMAForecaster,
    AutoARIMAForecaster,
    AutoETSForecaster,
    DummyForecaster,
    ETSForecaster,
    NaiveForecaster,
)
# from aeon.forecasting._sktime_autoets import SktimeAutoETSForecaster
# from aeon.forecasting._statsforecast_autoets import StatsForecastAutoETSForecaster

from tsml_eval.utils.functions import str_in_nested_list

stats_forecasters = [
    ["etsforecaster", "ets"],
    # ["autoetsforecaster", "autoets"],
    ["autosarima", "sarima"],
    ["autoarima", "arima"],
    # "sktimeets",
    # "statsforecastets",
]
other_forecasters = [
    ["dummyforecaster", "dummy"],
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

    if str_in_nested_list(stats_forecasters, f):
        return _set_forecaster_stats(f, random_state, n_jobs, kwargs)
    elif str_in_nested_list(other_forecasters, f):
        return _set_forecaster_other(f, random_state, n_jobs, kwargs)
    else:
        raise ValueError(f"UNKNOWN FORECASTER: {f} in get_forecaster_by_name")


def _set_forecaster_stats(f, random_state, n_jobs, kwargs):
    if f == "etsforecaster" or f == "ets":
        return ETSForecaster(**kwargs)
    # if f == "autoetsforecaster" or f == "autoets":
    #     return AutoETSForecaster(**kwargs)
    # if f == "sktimeets":
    #     return SktimeAutoETSForecaster(**kwargs)
    # if f == "statsforecastets":
    #     return StatsForecastAutoETSForecaster(**kwargs)
    if f == "autosarima" or f == "sarima":
        return AutoSARIMAForecaster(**kwargs)
    if f == "autoarima" or f == "arima":
        return AutoARIMAForecaster(**kwargs)


def _set_forecaster_other(f, random_state, n_jobs, kwargs):
    if f == "dummyforecaster" or f == "dummy":
        return DummyForecaster(**kwargs)
    if f == "naiveforecaster" or f == "naive":
        return NaiveForecaster(**kwargs)
