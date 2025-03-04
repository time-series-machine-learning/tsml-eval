"""Set forecaster function."""

__author__ = ["MatthewMiddlehurst"]

from aeon.forecasting.naive import NaiveForecaster

from tsml_eval.utils.functions import str_in_nested_list

# ml_forecasters = []
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

    # if str_in_nested_list(ml_forecasters, f):
    #     return _set_forecaster_ml(f, random_state, n_jobs, kwargs)
    if str_in_nested_list(other_forecasters, f):
        return _set_forecaster_other(f, random_state, n_jobs, kwargs)
    else:
        raise ValueError(f"UNKNOWN FORECASTER: {f} in set_forecaster")


# def _set_forecaster_ml(f, random_state, n_jobs, kwargs):
#     pass


def _set_forecaster_other(f, random_state, n_jobs, kwargs):
    if f == "naiveforecaster" or f == "naive":
        return NaiveForecaster(**kwargs)
