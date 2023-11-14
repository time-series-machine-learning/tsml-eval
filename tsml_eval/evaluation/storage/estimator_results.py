"""Abstract class for storing and loading results from an experiment."""

from abc import ABC, abstractmethod


class EstimatorResults(ABC):
    """
    Abstract base class for storing estimator results.

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset.
    estimator_name : str, optional
        Name of the estimator.
    split : str, optional
        Dataset split (e.g., 'train' or 'test').
    resample_id : int, optional
        Identifier for the data fold.
    time_unit : str, optional
        Unit of time measurement, default is "nanoseconds".
    description : str, optional
        A human-friendly description of the estimator results.
    parameters : str, optional
        Estimator parameters and other related information as a string.
    fit_time : float, optional
        Time taken to build the estimator.
    predict_time : float, optional
        Time taken to test the estimator.
    benchmark_time : float, optional
        Time taken to benchmark the estimator.
    memory_usage : float, optional
        Memory usage of the estimator.

    """

    def __init__(
        self,
        dataset_name="N/A",
        estimator_name="N/A",
        split="N/A",
        resample_id=-1,
        time_unit="nanoseconds",
        description="",
        parameters="No parameter info",
        fit_time=-1.0,
        predict_time=-1.0,
        benchmark_time=-1.0,
        memory_usage=-1.0,
    ):
        # Line 1
        self.dataset_name = dataset_name
        self.estimator_name = estimator_name
        self.split = split
        self.resample_id = resample_id
        self.time_unit = time_unit
        self.description = description

        # Line 2
        self.parameter_info = parameters

        # Line 3
        self.fit_time = fit_time
        self.predict_time = predict_time
        self.benchmark_time = benchmark_time
        self.memory_usage = memory_usage

        self.build_time_milli_ = None
        self.median_pred_time_milli_ = None

    # var_name: (display_name, higher is better)
    statistics = {
        "fit_time": ("FitTime", False),
        "predict_time": ("PredictTime", False),
        "memory_usage": ("MemoryUsage", False),
    }

    @abstractmethod
    def save_to_file(self, file_path):
        """Save results to a specified file.

        Parameters
        ----------
        file_path : str
            The path to the file where the results will be saved.
        """
        pass

    @abstractmethod
    def load_from_file(self, file_path):
        """Load results from a specified file.

        Parameters
        ----------
        file_path : str
            The path to the file where the results will be loaded from.
        """
        pass

    @abstractmethod
    def calculate_statistics(self, overwrite=False):
        """Calculate statistics from the results.

        This method should handle any necessary calculations to produce statistics
        from the results data held within the object.
        """
        pass
