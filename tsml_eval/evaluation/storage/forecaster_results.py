"""Class for storing and loading results from a forecasting experiment."""

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

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
    parameters : str, default="No parameter info"
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
        parameters="No parameter info",
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

        super().__init__(
            dataset_name=dataset_name,
            estimator_name=forecaster_name,
            split=split,
            resample_id=random_seed,
            time_unit=time_unit,
            description=description,
            parameters=parameters,
            fit_time=fit_time,
            predict_time=predict_time,
            benchmark_time=benchmark_time,
            memory_usage=memory_usage,
        )

    # var_name: (display_name, higher is better, is timing)
    statistics = {
        "mean_absolute_percentage_error": ("MAPE", False, False),
        **EstimatorResults.statistics,
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
        parameters=lines[1].strip(),
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
