"""Abstract class for storing and loading results from an experiment."""

from abc import ABC, abstractmethod


class EstimatorResults(ABC):
    """
    Abstract class for storing and loading results from an experiment.

    Parameters
    ----------
    dataset_name : str, default="N/A"
        Name of the dataset used.
    estimator_name : str, default="N/A"
        Name of the estimator used.
    split : str, default="N/A"
        Type of data split used, i.e. "train" or "test".
    resample_id : int or None, default=None
        Random seed used for the data resample, with 0 usually being the original data.
    time_unit : str, default="nanoseconds"
        Time measurement used for other fields.
    description : str, default=""
        Additional description of the experiment. Appended to the end
        of the first line of the results file.
    parameters : str, default="No parameter info"
        Information about parameters used in the estimator and other build information.
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

    # var_name: (display_name, higher is better, is timing)
    statistics = {
        "fit_time": ("FitTime", False, True),
        "predict_time": ("PredictTime", False, True),
        "memory_usage": ("MemoryUsage", False, False),
    }

    @abstractmethod
    def save_to_file(self, file_path, full_path=True):
        """
        Write the estimator results into a file format used by tsml.

        Abstract, must be implemented by subclasses.

        Parameters
        ----------
        file_path : str
            Path to write the results file to or the directory to build the default file
            structure if full_path is False.
        full_path : boolean, default=True
            If True, results are written directly to the directory passed in file_path.
            If False, then a standard file structure using the estimator and dataset
            names is created and used to write the results file.
        """
        pass

    @abstractmethod
    def load_from_file(self, file_path, verify_values=True):
        """
        Load estimator results from a specified file.

        This method reads a file containing estimator results and reconstructs the
        EstimatorResults object. It calculates performance statistics and
        verifies values based on the loaded data.

        Abstract, must be implemented by subclasses.

        Parameters
        ----------
        file_path : str
            The path to the file from which estimator results should be loaded. The
            file should be a tsml formatted estimator results file.
        verify_values : bool, default=True
            If the method should perform verification of the loaded values.


        Returns
        -------
        self : EstimatorResults
            The same EstimatorResults object with loaded results.
        """
        pass

    @abstractmethod
    def calculate_statistics(self, overwrite=False):
        """
        Calculate various performance statistics based on the estimator results.

        This method computes various performance metrics based on the estimators output.

        Abstract, must be implemented by subclasses.

        Parameters
        ----------
        overwrite : bool, default=False
            If the function should overwrite the current values when they are not None.
        """
        pass
