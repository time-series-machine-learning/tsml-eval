"""Class for storing and loading results from a forecasting experiment."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from tsml_eval.evaluation.storage.estimator_results import EstimatorResults
from tsml_eval.utils.results_writing import write_forecasting_results


class ForecasterResults(EstimatorResults):
    """
    A class for storing and managing results from forecasting experiments.

    This class provides functionalities for storing forecaster results,
    including predictions, probabilities, and various performance metrics.
    It extends the `EstimatorResults` class, inheriting its base functionalities.

    Parameters
    ----------
    dataset_name : str, default="N/A"
        Name of the dataset used.
    forecaster_name : str, default="N/A"
        Name of the forecaster used.
    split : str, default="N/A"
        Type of data split used, i.e. "train" or "test".
    random_seed : int or None, default=None
        Random seed used.
    time_unit : str, default="nanoseconds"
        Time measurement used for other fields.
    description : str, default=""
        Additional description of the forecasting experiment. Appended to the end
        of the first line of the results file.
    parameter_info : str, default="No parameter info"
        Information about parameters used in the forecaster and other build information.
        Written to the second line of the results file.
    fit_time : float, default=-1.0
        Time taken fitting the model.
    predict_time : float, default=-1.0
        Time taken making predictions.
    benchmark_time : float, default=-1.0
        Time taken to run a simple benchmark function. In tsml-eval experiments, this
        is the time spent to sort 1,000 (seeded) random numpy arrays of size 20,000.
    memory_usage : float, default=-1.0
        Memory usage during the experiment. In tsml-eval experiments, this is the peak
        memory usage during the fit method.
    target_labels : array-like or None, default=None
        Actual target labels.
    predictions : array-like or None, default=None
        Predicted target labels.
    pred_times : array-like or None, default=None
        Prediction times for each case.
    pred_descriptions : list of str or None, default=None
        Descriptions for each prediction.

    Attributes
    ----------
    mean_absolute_percentage_error : float or None
        Mean absolute percentage error of the predictions.

    Examples
    --------
    >>> from tsml_eval.evaluation.storage import ForecasterResults
    >>> from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH
    >>> fr = ForecasterResults().load_from_file(
    ...     _TEST_RESULTS_PATH +
    ...     "/forecasting/NaiveForecaster/Predictions/Airline/testResample0.csv"
    ... )
    >>> fr.calculate_statistics()
    >>> mape = fr.mean_absolute_percentage_error
    """

    def __init__(
        self,
        dataset_name="N/A",
        forecaster_name="N/A",
        split="N/A",
        random_seed=None,
        time_unit="nanoseconds",
        description="",
        parameter_info="No parameter info",
        fit_time=-1.0,
        predict_time=-1.0,
        benchmark_time=-1.0,
        memory_usage=-1.0,
        target_labels=None,
        predictions=None,
        pred_times=None,
        pred_descriptions=None,
    ):
        # Line 1
        self.forecaster_name = forecaster_name
        self.random_seed = random_seed

        # Results
        self.target_labels = target_labels
        self.predictions = predictions
        self.pred_times = pred_times
        self.pred_descriptions = pred_descriptions

        self.forecasting_horizon = None

        self.mean_absolute_percentage_error = None
        self.symmetric_mean_absolute_percentage_error = None
        self.mean_absolute_error = None
        self.root_mean_squared_error = None
        self.naive_scaled_mean_absolute_error = None

        # Official M4 statistics. Only computed when a training series context is
        # attached via set_m4_context (e.g. by passing data_path to
        # evaluate_forecasters_by_problem); None otherwise, and the OWA statistic is
        # dropped from evaluations that do not have it.
        self.m4_mean_absolute_scaled_error = None
        self.naive2_symmetric_mean_absolute_percentage_error = None
        self.naive2_mean_absolute_scaled_error = None
        self.overall_weighted_average = None
        self._m4_train = None
        self._m4_seasonal_period = None

        super().__init__(
            dataset_name=dataset_name,
            estimator_name=forecaster_name,
            split=split,
            resample_id=random_seed,
            time_unit=time_unit,
            description=description,
            parameter_info=parameter_info,
            fit_time=fit_time,
            predict_time=predict_time,
            benchmark_time=benchmark_time,
            memory_usage=memory_usage,
        )

    # var_name: (display_name, higher is better, is timing)
    # To re-enable timing/memory diagrams, add back to _DISABLED_STATISTICS:
    #   "fit_time", "predict_time", "memory_usage"
    statistics = {
        "mean_absolute_percentage_error": ("MAPE", False, False),
        "symmetric_mean_absolute_percentage_error": ("sMAPE", False, False),
        "mean_absolute_error": ("MAE", False, False),
        "root_mean_squared_error": ("RMSE", False, False),
        "naive_scaled_mean_absolute_error": ("MASE", False, False),
        "overall_weighted_average": ("OWA", False, False),
        **{
            k: v
            for k, v in EstimatorResults.statistics.items()
            if k not in {"fit_time", "predict_time", "memory_usage"}
        },
    }

    def save_to_file(self, file_path, full_path=True):
        """
        Write the forecaster results into a file format used by tsml.

        Parameters
        ----------
        file_path : str
            Path to write the results file to or the directory to build the default file
            structure if full_path is False.
        full_path : boolean, default=True
            If True, results are written directly to the directory passed in file_path.
            If False, then a standard file structure using the forecaster and dataset
            names is created and used to write the results file.
        """
        self.infer_size()

        if self.mean_absolute_percentage_error is None:
            self.mean_absolute_percentage_error = mean_absolute_percentage_error(
                self.target_labels, self.predictions
            )

        write_forecasting_results(
            self.predictions,
            self.target_labels,
            self.estimator_name,
            self.dataset_name,
            file_path,
            full_path=full_path,
            split=self.split,
            random_seed=self.resample_id,
            time_unit=self.time_unit,
            first_line_comment=self.description,
            parameter_info=self.parameter_info,
            mape=self.mean_absolute_percentage_error,
            fit_time=self.fit_time,
            predict_time=self.predict_time,
            benchmark_time=self.benchmark_time,
            memory_usage=self.memory_usage,
        )

    def load_from_file(self, file_path, verify_values=True):
        """
        Load forecaster results from a specified file.

        This method reads a file containing forecaster results and reconstructs the
        ForecasterResults object. It calculates performance statistics and
        verifies values based on the loaded data.

        Parameters
        ----------
        file_path : str
            The path to the file from which forecaster results should be loaded. The
            file should be a tsml formatted forecaster results file.
        verify_values : bool, default=True
            If the method should perform verification of the loaded values.

        Returns
        -------
        self : ForecasterResults
            The same ForecasterResults object with loaded results.
        """
        fr = load_forecaster_results(file_path, verify_values=verify_values)
        self.__dict__.update(fr.__dict__)
        return self

    def calculate_statistics(self, overwrite=False):
        """
        Calculate various performance statistics based on the forecaster results.

        This method computes various performance metrics, such as MAPE based on the
        forecasters output.

        Parameters
        ----------
        overwrite : bool, default=False
            If the function should overwrite the current values when they are not None.
        """
        self.infer_size(overwrite=overwrite)

        if self.mean_absolute_percentage_error is None or overwrite:
            self.mean_absolute_percentage_error = mean_absolute_percentage_error(
                self.target_labels, self.predictions
            )

        if self.symmetric_mean_absolute_percentage_error is None or overwrite:
            denom = np.abs(self.target_labels) + np.abs(self.predictions)
            # treat both-zero as zero error
            smape_vals = np.where(
                denom == 0, 0.0, np.abs(self.target_labels - self.predictions) / denom
            )
            self.symmetric_mean_absolute_percentage_error = float(
                np.mean(smape_vals) * 2
            )

        if self.mean_absolute_error is None or overwrite:
            self.mean_absolute_error = float(
                mean_absolute_error(self.target_labels, self.predictions)
            )

        if self.root_mean_squared_error is None or overwrite:
            self.root_mean_squared_error = float(
                np.sqrt(np.mean((self.target_labels - self.predictions) ** 2))
            )

        if self._m4_train is not None and (
            self.overall_weighted_average is None or overwrite
        ):
            self._calculate_m4_statistics()

        if self.naive_scaled_mean_absolute_error is None or overwrite:
            # Approximates MASE using the test set. Denominator is the mean absolute
            # step change in the test actuals (lag-1 naive MAE on test set).
            # Falls back to mean(|y[t]|) if the series is constant, and 0.0 if all
            # actuals are zero. Values <1 mean the model beats naive; >1 means it does not.
            naive_mae = np.mean(np.abs(np.diff(self.target_labels))) if len(self.target_labels) > 1 else 0.0
            if naive_mae == 0:
                naive_mae = np.mean(np.abs(self.target_labels))
            self.naive_scaled_mean_absolute_error = float(
                self.mean_absolute_error / naive_mae
            ) if naive_mae != 0 else 0.0

    def set_m4_context(self, train, seasonal_period):
        """Attach the training series needed for official M4 statistics (OWA).

        The M4 competition ranked entries by OWA, which requires the training series:
        MASE is scaled by the in-sample seasonal naive error, and both sMAPE and MASE
        are expressed relative to the Naive2 benchmark (seasonally adjusted naive).
        Neither can be derived from the stored predictions alone, so these statistics
        are only computed when this context is attached.

        Parameters
        ----------
        train : np.ndarray
            The training series the forecaster was fit on, i.e. the full series minus
            the ``len(target_labels)`` test values.
        seasonal_period : int
            The M4 frequency used for MASE scaling and Naive2 (yearly=1, quarterly=4,
            monthly=12, weekly=1, daily=1, hourly=24 in the original competition).
        """
        self._m4_train = np.asarray(train, dtype=float)
        self._m4_seasonal_period = int(seasonal_period)
        self.overall_weighted_average = None

    def _calculate_m4_statistics(self):
        """Compute official M4 MASE, Naive2 baselines and per-series OWA."""
        train = self._m4_train
        ppy = self._m4_seasonal_period
        y = np.asarray(self.target_labels, dtype=float)
        preds = np.asarray(self.predictions, dtype=float)

        # Official M4 MASE scales by the in-sample seasonal naive MAE, using the
        # frequency unconditionally (no seasonality test).
        m = ppy if 0 < ppy < len(train) else 1
        mase_denominator = np.mean(np.abs(train[m:] - train[:-m]))

        naive2 = _naive2_forecast(train, len(y), ppy)
        self.naive2_symmetric_mean_absolute_percentage_error = _smape(y, naive2)

        if mase_denominator == 0:
            self.m4_mean_absolute_scaled_error = np.nan
            self.naive2_mean_absolute_scaled_error = np.nan
        else:
            self.m4_mean_absolute_scaled_error = float(
                np.mean(np.abs(y - preds)) / mase_denominator
            )
            self.naive2_mean_absolute_scaled_error = float(
                np.mean(np.abs(y - naive2)) / mase_denominator
            )

        model_smape = _smape(y, preds)
        if (
            self.naive2_symmetric_mean_absolute_percentage_error == 0
            or self.naive2_mean_absolute_scaled_error == 0
            or not np.isfinite(self.m4_mean_absolute_scaled_error)
            or not np.isfinite(self.naive2_mean_absolute_scaled_error)
        ):
            self.overall_weighted_average = np.nan
        else:
            self.overall_weighted_average = float(
                0.5
                * (
                    model_smape
                    / self.naive2_symmetric_mean_absolute_percentage_error
                    + self.m4_mean_absolute_scaled_error
                    / self.naive2_mean_absolute_scaled_error
                )
            )

    def infer_size(self, overwrite=False):
        """
        Infer and return the size of the dataset used in the results.

        This method estimates the size of the dataset that was used for the estimator,
        based on the results data.

        Parameters
        ----------
        overwrite : bool, default=False
            If the function should overwrite the current values when they are not None.
        """
        if self.forecasting_horizon is None or overwrite:
            self.forecasting_horizon = len(self.target_labels)


def _smape(y_true, y_pred):
    """Symmetric MAPE, as a fraction (multiply by 100 for the M4 percentage form)."""
    denom = np.abs(y_true) + np.abs(y_pred)
    vals = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)
    return float(np.mean(vals) * 2)


def _acf(x, max_lag):
    """Autocorrelation for lags 1..max_lag, R acf-style (biased, overall mean)."""
    n = len(x)
    x_bar = np.mean(x)
    denominator = np.sum((x - x_bar) ** 2)
    if denominator == 0:
        return np.zeros(max_lag)
    return np.array(
        [
            np.sum((x[: n - lag] - x_bar) * (x[lag:] - x_bar)) / denominator
            for lag in range(1, max_lag + 1)
        ]
    )


def _seasonality_test(x, ppy):
    """M4 benchmark seasonality test: 90% significance autocorrelation at lag ppy."""
    if ppy <= 1 or len(x) < 3 * ppy:
        return False
    r = _acf(x, ppy)
    limit = 1.645 / np.sqrt(len(x)) * np.sqrt(1 + 2 * np.sum(r[:-1] ** 2))
    result = np.abs(r[-1]) > limit
    return bool(result) if np.isfinite(r[-1]) else False


def _seasonal_indices(x, ppy):
    """Seasonal indices from classical multiplicative decomposition (R decompose).

    Returns indices for seasonal positions 0..ppy-1, where the position of x[t] is
    t % ppy, normalised to mean 1.
    """
    if ppy % 2 == 0:
        weights = np.concatenate(([0.5], np.ones(ppy - 1), [0.5])) / ppy
    else:
        weights = np.ones(ppy) / ppy
    trend = np.convolve(x, weights, mode="valid")
    offset = (len(weights) - 1) // 2
    with np.errstate(divide="ignore", invalid="ignore"):
        detrended = x[offset : offset + len(trend)] / trend
    positions = (offset + np.arange(len(detrended))) % ppy
    indices = np.array(
        [np.nanmean(np.where(positions == s, detrended, np.nan)) for s in range(ppy)]
    )
    return indices / np.mean(indices)


def _naive2_forecast(train, horizon, ppy):
    """The M4 Naive2 benchmark: naive on the seasonally adjusted series.

    If the series tests as seasonal at ppy, it is deseasonalised via classical
    multiplicative decomposition, the last value is repeated over the horizon, and the
    forecasts are reseasonalised. Otherwise this is a plain naive forecast.
    """
    train = np.asarray(train, dtype=float)
    if _seasonality_test(train, ppy):
        indices = _seasonal_indices(train, ppy)
        with np.errstate(divide="ignore", invalid="ignore"):
            deseasonalised = train / indices[np.arange(len(train)) % ppy]
        out_indices = indices[(len(train) + np.arange(horizon)) % ppy]
        return deseasonalised[-1] * out_indices
    return np.full(horizon, train[-1])


def load_forecaster_results(file_path, calculate_stats=True, verify_values=True):
    """
    Load and return forecaster results from a specified file.

    This function reads a file containing forecaster results and reconstructs the
    ForecasterResults object. It optionally calculates performance statistics and
    verifies values based on the loaded data.

    Parameters
    ----------
    file_path : str
        The path to the file from which forecaster results should be loaded. The file
        should be a tsml formatted forecaster results file.
    calculate_stats : bool, default=True
        Whether to calculate performance statistics from the loaded results.
    verify_values : bool, default=True
        If the function should perform verification of the loaded values.

    Returns
    -------
    fr : ForecasterResults
        A ForecasterResults object containing the results loaded from the file.
    """
    with open(file_path) as file:
        lines = file.readlines()

        line1 = lines[0].split(",")
        line3 = lines[2].split(",")
        mape = float(line3[0])
        fh = len(lines) - 3

        line_size = len(lines[3].split(","))

        target_labels = np.zeros(fh)
        predictions = np.zeros(fh)

        if line_size > 3:
            pred_times = np.zeros(fh)
        else:
            pred_times = None

        if line_size > 5:
            pred_descriptions = []
        else:
            pred_descriptions = None

        for i in range(0, fh):
            line = lines[i + 3].split(",")
            target_labels[i] = float(line[0])
            predictions[i] = float(line[1])

            if pred_times is not None:
                pred_times[i] = float(line[3])

            if pred_descriptions is not None:
                pred_descriptions.append(",".join(line[5:]).strip())

    fr = ForecasterResults(
        dataset_name=line1[0],
        forecaster_name=line1[1],
        split=line1[2],
        random_seed=None if line1[3] == "None" else int(line1[3]),
        time_unit=line1[4].lower(),
        description=",".join(line1[5:]).strip(),
        parameter_info=lines[1].strip(),
        fit_time=float(line3[1]),
        predict_time=float(line3[2]),
        benchmark_time=float(line3[3]),
        memory_usage=float(line3[4]),
        target_labels=target_labels,
        predictions=predictions,
        pred_times=pred_times,
        pred_descriptions=pred_descriptions,
    )

    if calculate_stats:
        fr.calculate_statistics()

    if verify_values:
        fr.infer_size(overwrite=True)
        assert fr.forecasting_horizon == fh

        if calculate_stats:
            assert fr.mean_absolute_percentage_error == mape

    return fr
