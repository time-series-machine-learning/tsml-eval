"""Class for storing and loading results from a regression experiment."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

import tsml_eval.evaluation.storage as storage
from tsml_eval.evaluation.storage.estimator_results import EstimatorResults
from tsml_eval.utils.experiments import write_regression_results


class RegressorResults(EstimatorResults):
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

        super(RegressorResults, self).__init__(
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

    # var_name: (display_name, higher is better)
    statistics = {
        "mean_squared_error": ("MSE", False),
        "root_mean_squared_error": ("RMSE", False),
        "mean_absolute_error": ("MAE", False),
        "r2_score": ("R2", True),
        "mean_absolute_percentage_error": ("MAPE", False),
        **EstimatorResults.statistics,
    }

    def save_to_file(self, file_path, full_path=True):
        """
        Writes the full results to a file.

        Parameters
        ----------
        file_path : str
            The path of the file to write the results to.
        full_path : boolean, default=True
            If True, results are written directly to the directory passed in output_path.
            If False, then a standard file structure using the classifier and dataset names
            is created and used to write the results file.
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

    def load_from_file(self, file_path, calculate_stats=True, verify_values=True):
        """Load results from a specified file.

        Parameters
        ----------
        file_path : str
            The path to the file where the results will be loaded from.
        """
        rr = storage.load_regressor_results(file_path)
        self.__dict__.update(rr.__dict__)
        return self

    def calculate_statistics(self, overwrite=False):
        """Calculate statistics from the results.

        This method should handle any necessary calculations to produce statistics
        from the results data held within the object.
        """
        self.infer_size(overwrite=overwrite)

        if self.mean_squared_error is None or overwrite:
            self.mean_squared_error = mean_squared_error(
                self.target_labels, self.predictions
            )
        if self.root_mean_squared_error is None or overwrite:
            self.root_mean_squared_error = mean_squared_error(
                self.target_labels, self.predictions, squared=False
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
        if self.n_cases is None or overwrite:
            self.n_cases = len(self.target_labels)


def load_regressor_results(file_path, calculate_stats=True, verify_values=True):
    with open(file_path, "r") as file:
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
                pred_times[i] = float(line[4])

            if pred_descriptions is not None:
                pred_descriptions.append(",".join(line[6]).strip())

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
