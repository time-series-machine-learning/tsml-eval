"""DDrCIF: DrCIF regression forecaster applied to differenced data.

DrCIF (as a regressor wrapped in a window-based forecaster) tends to perform
better on differenced, stationary data. ``DDrCIF`` is a pipeline forecaster that

1. differences the input series (first order by default) using aeon's
   :class:`~aeon.transformations.series._diff.DifferenceTransformer`,
2. fits a window-based :class:`~aeon.forecasting.RegressionForecaster` using a
   :class:`~aeon.regression.interval_based.DrCIFRegressor` on the differenced
   series to forecast the *next difference*, and
3. reconstructs (undifferences) that forecast back into the original scale.
"""

import numpy as np
from aeon.forecasting import BaseForecaster, RegressionForecaster
from aeon.forecasting.utils._undifference import _undifference
from aeon.transformations.series._diff import DifferenceTransformer

__all__ = ["DDrCIF"]


class DDrCIF(BaseForecaster):
    """Differenced DrCIF forecaster.

    Forecasts one step ahead by running a :class:`DrCIFRegressor` (wrapped in a
    window-based :class:`RegressionForecaster`) on the ``order``-th differenced
    series and reconstructing the prediction back to the original scale.

    Parameters
    ----------
    window : int, default=100
        Window length used by the underlying ``RegressionForecaster``. This is the
        number of prior (differenced) points used to predict the next one.
    order : int, default=1
        Order of differencing applied to the series before forecasting. The
        forecast is undifferenced by the same order.
    n_estimators : int, default=200
        Number of trees in the ``DrCIFRegressor``.
    random_state : int, RandomState instance or None, default=None
        Random seed or ``RandomState`` object for the ``DrCIFRegressor``.
    n_jobs : int, default=1
        Number of jobs to run in parallel for the ``DrCIFRegressor``.

    Notes
    -----
    The reconstruction from the differenced forecast is delegated to aeon's
    :func:`~aeon.forecasting.utils._undifference._undifference`. Feeding it the
    single forecast difference together with the last ``order`` observed values
    yields the next original-scale value, which for the default first order
    reduces to ``y[t] = p + y[t-1]``.
    """

    _tags = {
        "capability:horizon": False,  # one step ahead only
    }

    def __init__(
        self,
        window=100,
        order=1,
        n_estimators=200,
        random_state=None,
        n_jobs=1,
    ):
        self.window = window
        self.order = order
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        from aeon.regression.interval_based import DrCIFRegressor

        self.differencer_ = DifferenceTransformer(order=self.order)
        y_diff = self.differencer_.fit_transform(y)

        regressor = DrCIFRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            parallel_backend="loky",
        )
        self.forecaster_ = RegressionForecaster(
            window=self.window, regressor=regressor
        )
        self.forecaster_.fit(y_diff)

        # Reconstruct the one-step-ahead forecast for the `forecast` path.
        self.forecast_ = self._reconstruct(y, self.forecaster_.forecast_)
        return self

    def _predict(self, y, exog=None):
        y_diff = self.differencer_.transform(y)
        diff_pred = self.forecaster_.predict(y_diff)
        return self._reconstruct(y, diff_pred)

    def _forecast(self, y, exog=None):
        """Fit and forecast one step ahead for time series y."""
        self._fit(y, exog)
        return self.forecast_

    def _reconstruct(self, y, diff_pred):
        """Undifference a single forecast difference back to the original scale.

        Delegates to aeon's ``_undifference``, passing the forecast difference as a
        length-one differenced series and the last ``order`` observed values of ``y``
        as the initial values, then returns the final (next) reconstructed value.
        """
        y = np.asarray(y).ravel()
        initial_values = y[-self.order :].astype(float)
        reconstructed = _undifference(np.array([float(diff_pred)]), initial_values)
        return reconstructed[-1]
