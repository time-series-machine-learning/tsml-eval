"""Class for storing and loading results from a regression experiment."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from tsml_eval.evaluation.storage.estimator_results import EstimatorResults
from tsml_eval.utils.results_writing import write_regression_results


class RegressorResults(EstimatorResults):
    """
    A class for storing and managing results from regression experiments.

    This class provides functionalities for storing regressor results,
    including predictions, probabilities, and various performance metrics.
    It extends the `EstimatorResults` class, inheriting its base functionalities.

    Parameters
    ----------
    dataset_name : str, default="N/A"
        Name of the dataset used.
    regressor_name : str, default="N/A"
        Name of the regressor used.
    split : str, default="N/A"
        Type of data split used, i.e. "train" or "test".
    resample_id : int or None, default=None
        Random seed used for the data resample, with 0 usually being the original data.
    time_unit : str, default="nanoseconds"
        Time measurement used for other fields.
    description : str, default=""
        Additional description of the regression experiment. Appended to the end
        of the first line of the results file.
    parameters : str, default="No parameter info"
        Information about parameters used in the regressor and other build information.
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
    error_estimate_method : str, default="N/A"
        Method used for train error/accuracy estimation (i.e. 10-fold CV, OOB error).
    error_estimate_time : float, default=-1.0
        Time taken for train error/accuracy estimation.
    build_plus_estimate_time : float, default=-1.0
        Total time for building the regressor and estimating error/accuracy on the
        train set. For certain methods this can be different from the sum of fit_time
        and error_estimate_time.
    target_labels : array-like or None, default=None
        Actual target labels.
    predictions : array-like or None, default=None
        Predicted class labels.
    pred_times : array-like or None, default=None
        Prediction times for each case.
    pred_descriptions : list of str or None, default=None
        Descriptions for each prediction.

    Attributes
    ----------
    n_cases : int or None
        Number of cases in the dataset.
    mean_squared_error : float or None
        Mean squared error of the predictions.
    root_mean_squared_error : float or None
        Root mean squared error of the predictions.
    mean_absolute_error : float or None
        Mean absolute error of the predictions.
    r2_score : float or None
        R2 score of the predictions.
    mean_absolute_percentage_error : float or None
        Mean absolute percentage error of the predictions.

    Examples
    --------
    >>> from tsml_eval.evaluation.storage import RegressorResults
    >>> from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH
    >>> rr = RegressorResults().load_from_file(
    ...     _TEST_RESULTS_PATH +
    ...     "/regression/ROCKET/Predictions/Covid3Month/testResample0.csv"
    ... )
    >>> rr.calculate_statistics()
    >>> mse = rr.mean_squared_error
    """

    def __init__(
        self,
        dataset_name="N/A",
        regressor_name="N/A",
        split="N/A",
        resample_id=None,
        time_unit="nanoseconds",
        description="",
        parameters="No parameter info",
        fit_time=-1.0,
        predict_time=-1.0,
        benchmark_time=-1.0,
        memory_usage=-1.0,
        error_estimate_method="N/A",
        error_estimate_time=-1.0,
        build_plus_estimate_time=-1.0,
        target_labels=None,
        predictions=None,
        pred_times=None,
        pred_descriptions=None,
    ):
        # Line 1
        self.regressor_name = regressor_name

        # Line 3
        self.train_estimate_method = error_estimate_method
        self.train_estimate_time = error_estimate_time
        self.fit_and_estimate_time = build_plus_estimate_time

        # Results
        self.target_labels = target_labels
        self.predictions = predictions
        self.pred_times = pred_times
        self.pred_descriptions = pred_descriptions

        self.n_cases = None

        self.mean_squared_error = None
        self.root_mean_squared_error = None
        self.mean_absolute_error = None
        self.r2_score = None
        self.mean_absolute_percentage_error = None

        super().__init__(
            dataset_name=dataset_name,
            estimator_name=regressor_name,
            split=split,
            resample_id=resample_id,
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
        "mean_squared_error": ("MSE", False, False),
        "root_mean_squared_error": ("RMSE", False, False),
        "mean_absolute_error": ("MAE", False, False),
        "r2_score": ("R2", True, False),
        "mean_absolute_percentage_error": ("MAPE", False, False),
        **EstimatorResults.statistics,
    }

    def save_to_file(self, file_path, full_path=True):
        """
        Write the regressor results into a file format used by tsml.

        Parameters
        ----------
        file_path : str
            Path to write the results file to or the directory to build the default file
            structure if full_path is False.
        full_path : boolean, default=True
            If True, results are written directly to the directory passed in file_path.
            If False, then a standard file structure using the regressor and dataset
            names is created and used to write the results file.
        """
        self.infer_size()

        if self.mean_squared_error is None:
            self.mean_squared_error = mean_squared_error(
                self.target_labels, self.predictions
            )

        write_regression_results(
            self.predictions,
            self.target_labels,
            self.estimator_name,
            self.dataset_name,
            file_path,
            full_path=full_path,
            split=self.split,
            resample_id=self.resample_id,
            time_unit=self.time_unit,
            first_line_comment=self.description,
            parameter_info=self.parameter_info,
            mse=self.mean_squared_error,
            fit_time=self.fit_time,
            predict_time=self.predict_time,
            benchmark_time=self.benchmark_time,
            memory_usage=self.memory_usage,
            train_estimate_method=self.train_estimate_method,
            train_estimate_time=self.train_estimate_time,
            fit_and_estimate_time=self.fit_and_estimate_time,
        )

    def load_from_file(self, file_path, verify_values=True):
        """
        Load regressor results from a specified file.

        This method reads a file containing regressor results and reconstructs the
        RegressorResults object. It calculates performance statistics and
        verifies values based on the loaded data.

        Parameters
        ----------
        file_path : str
            The path to the file from which regressor results should be loaded. The
            file should be a tsml formatted regressor results file.
        verify_values : bool, default=True
            If the method should perform verification of the loaded values.

        Returns
        -------
        self : RegressorResults
            The same RegressorResults object with loaded results.
        """
        rr = load_regressor_results(file_path, verify_values=verify_values)
        self.__dict__.update(rr.__dict__)
        return self

    def calculate_statistics(self, overwrite=False):
        """
        Calculate various performance statistics based on the regressor results.

        This method computes various performance metrics, such as MSE, MAPE,
        and others, based on the regressors output.

        Parameters
        ----------
        overwrite : bool, default=False
            If the function should overwrite the current values when they are not None.
        """
        self.infer_size(overwrite=overwrite)

        if self.mean_squared_error is None or overwrite:
            self.mean_squared_error = mean_squared_error(
                self.target_labels, self.predictions
            )
        if self.root_mean_squared_error is None or overwrite:
            self.root_mean_squared_error = root_mean_squared_error(
                self.target_labels, self.predictions
            )
        if self.mean_absolute_error is None or overwrite:
            self.mean_absolute_error = mean_absolute_error(
                self.target_labels, self.predictions
            )
        if self.r2_score is None or overwrite:
            self.r2_score = r2_score(self.target_labels, self.predictions)
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
        if self.n_cases is None or overwrite:
            self.n_cases = len(self.target_labels)


def load_regressor_results(file_path, calculate_stats=True, verify_values=True):
    """
    Load and return regressor results from a specified file.

    This function reads a file containing regressor results and reconstructs the
    RegressorResults object. It optionally calculates performance statistics and
    verifies values based on the loaded data.

    Parameters
    ----------
    file_path : str
        The path to the file from which regressor results should be loaded. The file
        should be a tsml formatted regressor results file.
    calculate_stats : bool, default=True
        Whether to calculate performance statistics from the loaded results.
    verify_values : bool, default=True
        If the function should perform verification of the loaded values.

    Returns
    -------
    rr : RegressorResults
        A RegressorResults object containing the results loaded from the file.
    """
    with open(file_path) as file:
        lines = file.readlines()

        line1 = lines[0].split(",")
        line3 = lines[2].split(",")
        mse = float(line3[0])
        n_cases = len(lines) - 3

        line_size = len(lines[3].split(","))

        target_labels = np.zeros(n_cases)
        predictions = np.zeros(n_cases)

        if line_size > 3:
            pred_times = np.zeros(n_cases)
        else:
            pred_times = None

        if line_size > 5:
            pred_descriptions = []
        else:
            pred_descriptions = None

        for i in range(0, n_cases):
            line = lines[i + 3].split(",")
            target_labels[i] = float(line[0])
            predictions[i] = float(line[1])

            if pred_times is not None:
                pred_times[i] = float(line[3])

            if pred_descriptions is not None:
                pred_descriptions.append(",".join(line[5:]).strip())

    rr = RegressorResults(
        dataset_name=line1[0],
        regressor_name=line1[1],
        split=line1[2],
        resample_id=None if line1[3] == "None" else int(line1[3]),
        time_unit=line1[4].lower(),
        description=",".join(line1[5:]).strip(),
        parameters=lines[1].strip(),
        fit_time=float(line3[1]),
        predict_time=float(line3[2]),
        benchmark_time=float(line3[3]),
        memory_usage=float(line3[4]),
        error_estimate_method=line3[5],
        error_estimate_time=float(line3[6]),
        build_plus_estimate_time=float(line3[7]),
        target_labels=target_labels,
        predictions=predictions,
        pred_times=pred_times,
        pred_descriptions=pred_descriptions,
    )

    if calculate_stats:
        rr.calculate_statistics()

    if verify_values:
        rr.infer_size(overwrite=True)
        assert rr.n_cases == n_cases

        if calculate_stats:
            assert rr.mean_squared_error == mse

    return rr
