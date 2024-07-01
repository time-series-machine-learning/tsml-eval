"""BaseForecaster, based on BaseSegmenter."""
from aeon.base._base_series import BaseSeriesEstimator
from abc import ABC, abstractmethod



class BaseForecaster(BaseSeriesEstimator, ABC):
    """Base class for all forecasting models.
    Parameters
    ----------
    axis : int
        Axis along which to segment if passed a multivariate series (2D input). If axis
        is 0, it is assumed each column is a time series and each row is a
        timepoint. i.e. the shape of the data is ``(n_timepoints,n_channels)``.
        ``axis == 1`` indicates the time series are in rows, i.e. the shape of the data
        is ``(n_channels, n_timepoints)`.
    forecast_horizon : int, default = 1
        Number of steps ahead to forecast.


    """
    def __init__(self, axis, forecast_horizon = 1):
        self.forecast_horizon = forecast_horizon
        self._is_fitted = False
        super().__init__(axis=axis)

    @final
    def fit(self, y, X=None, axis=1):
        """Fit forecaster to to y, possibly using exogenous data X.


        Parameters
        ----------
        X : one of aeon.base._base_series.VALID_INPUT_TYPES, default=None
            Further data to use in forecasting, must be aligned with y.
        y : one of aeon.base._base_series.VALID_INPUT_TYPES, default=None
            The training series to fit the forecaster on
        axis : int
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.

        Returns
        -------
        BaseForecaster
            The fitted estimator, reference to self.
        """
        # reset estimator at the start of fit
        self.reset()

        if X is not None:
            X = self._preprocess_series(X, axis, True)
        # y = self._check_y(y)
        self._fit(X=X, y=y)

        # this should happen last
        self._is_fitted = True
        return self

    @final
    def predict(self, X=None, y=None, axis=1) -> np.ndarray:
        """Forecast forecasting horizon ahead for train data or for y series.

        Parameters
        ----------
        X : one of aeon.base._base_series.VALID_INPUT_TYPES
            The time series to fit the model to.
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_INPUT_TYPES for aeon supported types.
        axis : int, default=1
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.

        Returns
        -------
        np.ndarray
            A boolean, int or float array of length len(X), where each element indicates
            whether the corresponding subsequence is anomalous or its anomaly score.
        """
        fit_empty = self.get_class_tag("fit_is_empty")
        if not fit_empty:
            self.check_is_fitted()

        X = self._preprocess_series(X, axis, fit_empty)

        return self._predict(X)

    @final
    def fit_predict(self,  y, X=None, axis=1) -> np.ndarray:
        """Fit forecaster using y and predict for y using the fitted model.

        Parameters
        ----------
        y : one of aeon.base._base_series.VALID_INPUT_TYPES
        X : one of aeon.base._base_series.VALID_INPUT_TYPES, default=None
        axis : int, default=1
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.

        Returns
        -------
        np.ndarray
            A boolean, int or float array of length len(X), where each element indicates
            whether the corresponding subsequence is anomalous or its anomaly score.
        """

        # reset estimator at the start of fit
        self.reset()
        if X is not None:
            X = self._preprocess_series(X, axis, True)
        pred = self._fit_predict(X, y)
        # this should happen last
        self._is_fitted = True
        return pred

    def _fit(self, X, y):
        return self

    @abstractmethod
    def _predict(self, X) -> np.ndarray: ...

    def _fit_predict(self, X, y):
        self._fit(X, y)
        return self._predict(X)


