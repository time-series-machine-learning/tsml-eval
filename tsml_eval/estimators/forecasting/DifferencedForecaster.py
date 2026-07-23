"""Generic differencing forecaster.

Some regressors (e.g. DrCIF) tend to perform better on differenced, stationary
data. :class:`DifferencedForecaster` is a generic pipeline forecaster that wraps
*any* regressor and

1. differences the input series (first order by default) using aeon's
   :class:`~aeon.transformations.series._diff.DifferenceTransformer`,
2. fits a window-based :class:`~aeon.forecasting.RegressionForecaster` using the
   supplied regressor on the differenced series to forecast the *next
   difference*, and
3. reconstructs (undifferences) that forecast back into the original scale.

This is the generic form of the old ``DDrCIF`` forecaster: ``DDrCIF`` is simply
``DifferencedForecaster`` with a :class:`DrCIFRegressor`.
"""

import numpy as np
from aeon.forecasting import BaseForecaster, RegressionForecaster
from aeon.forecasting.utils._undifference import _undifference
from aeon.transformations.series._diff import DifferenceTransformer

__all__ = ["DifferencedForecaster"]


class DifferencedForecaster(BaseForecaster):
    """Differenced regression forecaster.

    Forecasts one step ahead by running an arbitrary ``regressor`` (wrapped in a
    window-based :class:`RegressionForecaster`) on the ``order``-th differenced
    series and reconstructing the prediction back to the original scale.

    Parameters
    ----------
    regressor : object
        A regressor implementing the aeon/scikit-learn ``fit``/``predict``
        interface. It is wrapped in a ``RegressionForecaster`` and fitted on the
        differenced series.
    window : int, default=100
        Window length used by the underlying ``RegressionForecaster``. This is the
        number of prior (differenced) points used to predict the next one.
    order : int, default=1
        Order of differencing applied to the series before forecasting. The
        forecast is undifferenced by the same order.

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
        regressor,
        window=100,
        order=1,
    ):
        self.regressor = regressor
        self.window = window
        self.order = order
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        self.differencer_ = DifferenceTransformer(order=self.order)
        y_diff = self.differencer_.fit_transform(y)

        self.forecaster_ = RegressionForecaster(
            window=self.window, regressor=self.regressor
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
