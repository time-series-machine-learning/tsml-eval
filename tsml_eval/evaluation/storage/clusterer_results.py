"""Class for storing and loading results from a clustering experiment."""

import numpy as np
from aeon.benchmarking.metrics.clustering import clustering_accuracy_score
from numpy.testing import assert_allclose
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
)

from tsml_eval.evaluation.storage.estimator_results import EstimatorResults
from tsml_eval.utils.results_writing import write_clustering_results


class ClustererResults(EstimatorResults):
    """
    A class for storing and managing results from clustering experiments.

    This class provides functionalities for storing clustering results,
    including cluster labels, probabilities, and various performance metrics.
    It extends the `EstimatorResults` class, inheriting its base functionalities.

    Parameters
    ----------
    dataset_name : str, default="N/A"
        Name of the dataset used.
    clusterer_name : str, default="N/A"
        Name of the clusterer used.
    split : str, default="N/A"
        Type of data split used, i.e. "train" or "test".
    resample_id : int or None, default=None
        Random seed used for the data resample, with 0 usually being the original data.
    time_unit : str, default="nanoseconds"
        Time measurement used for other fields.
    description : str, default=""
        Additional description of the clustering experiment. Appended to the end
        of the first line of the results file.
    parameters : str, default="No parameter info"
        Information about parameters used in the clusterer and other build information.
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
    n_classes : int or None, default=None
        Number of classes in the dataset.
    n_clusters : int or None, default=None
        Number of clusters generated.
    class_labels : array-like or None, default=None
        Actual class labels.
    predictions : array-like or None, default=None
        Predicted cluster labels.
    probabilities : array-like or None, default=None
        Predicted cluster probabilities.
    pred_times : array-like or None, default=None
        Prediction times for each case.
    pred_descriptions : list of str or None, default=None
        Descriptions for each prediction.

    Attributes
    ----------
    n_cases : int or None
        Number of cases in the dataset.
    clustering_accuracy : float or None
        Clustering accuracy score.
    rand_index : float or None
        Rand score.
    adjusted_rand_index : float or None
        Adjusted Rand score.
    mutual_information : float or None
        Mutual information score.
    adjusted_mutual_information : float or None
        Adjusted mutual information score.
    normalised_mutual_information : float or None
        Normalised mutual information score.

    Examples
    --------
    >>> from tsml_eval.evaluation.storage import ClustererResults
    >>> from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH
    >>> cr = ClustererResults().load_from_file(
    ...     _TEST_RESULTS_PATH +
    ...     "/clustering/KMeans/Predictions/Trace/trainResample0.csv"
    ... )
    >>> cr.calculate_statistics()
    >>> acc = cr.clustering_accuracy
    """

    def __init__(
        self,
        dataset_name="N/A",
        clusterer_name="N/A",
        split="N/A",
        resample_id=None,
        time_unit="nanoseconds",
        description="",
        parameters="No parameter info",
        fit_time=-1.0,
        predict_time=-1.0,
        benchmark_time=-1.0,
        memory_usage=-1.0,
        n_classes=None,
        n_clusters=None,
        class_labels=None,
        predictions=None,
        probabilities=None,
        pred_times=None,
        pred_descriptions=None,
    ):
        # Line 1
        self.clusterer_name = clusterer_name

        # Line 3
        self.n_classes = n_classes
        self.n_clusters = n_clusters

        # Results
        self.class_labels = class_labels
        self.predictions = predictions
        self.probabilities = probabilities
        self.pred_times = pred_times
        self.pred_descriptions = pred_descriptions

        self.n_cases = None

        self.clustering_accuracy = None
        self.rand_index = None
        self.adjusted_rand_index = None
        self.mutual_information = None
        self.adjusted_mutual_information = None
        self.normalised_mutual_information = None

        super().__init__(
            dataset_name=dataset_name,
            estimator_name=clusterer_name,
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
        "clustering_accuracy": ("CLAcc", True, False),
        "rand_index": ("RI", True, False),
        "adjusted_rand_index": ("ARI", True, False),
        "mutual_information": ("MI", True, False),
        "adjusted_mutual_information": ("AMI", True, False),
        "normalised_mutual_information": ("NMI", True, False),
        **EstimatorResults.statistics,
    }

    def save_to_file(self, file_path, full_path=True):
        """
        Write the clusterer results into a file format used by tsml.

        Parameters
        ----------
        file_path : str
            Path to write the results file to or the directory to build the default file
            structure if full_path is False.
        full_path : boolean, default=True
            If True, results are written directly to the directory passed in file_path.
            If False, then a standard file structure using the clusterer and dataset
            names is created and used to write the results file.
        """
        self.infer_size()

        if self.clustering_accuracy is None:
            self.clustering_accuracy = clustering_accuracy_score(
                self.class_labels, self.predictions
            )

        write_clustering_results(
            self.predictions,
            self.probabilities,
            self.class_labels,
            self.estimator_name,
            self.dataset_name,
            file_path,
            full_path=full_path,
            split=self.split,
            resample_id=self.resample_id,
            time_unit=self.time_unit,
            first_line_comment=self.description,
            parameter_info=self.parameter_info,
            clustering_accuracy=self.clustering_accuracy,
            fit_time=self.fit_time,
            predict_time=self.predict_time,
            benchmark_time=self.benchmark_time,
            memory_usage=self.memory_usage,
            n_classes=self.n_classes,
            n_clusters=self.n_clusters,
        )

    def load_from_file(self, file_path, verify_values=True):
        """
        Load clusterer results from a specified file.

        This method reads a file containing clusterer results and reconstructs the
        ClustererResults object. It calculates performance statistics and
        verifies values based on the loaded data.

        Parameters
        ----------
        file_path : str
            The path to the file from which clusterer results should be loaded. The
            file should be a tsml formatted clusterer results file.
        verify_values : bool, default=True
            If the method should perform verification of the loaded values.

        Returns
        -------
        self : ClustererResults
            The same ClustererResults object with loaded results.
        """
        cr = load_clusterer_results(file_path, verify_values=verify_values)
        self.__dict__.update(cr.__dict__)
        return self

    def calculate_statistics(self, overwrite=False):
        """
        Calculate various performance statistics based on the clusterer results.

        This method computes various performance metrics, such as clustering accuracy,
        Rand score, and others, based on the clusterers output.

        Parameters
        ----------
        overwrite : bool, default=False
            If the function should overwrite the current values when they are not None.
        """
        self.infer_size(overwrite=overwrite)

        if self.clustering_accuracy is None or overwrite:
            self.clustering_accuracy = clustering_accuracy_score(
                self.class_labels, self.predictions
            )
        if self.rand_index is None or overwrite:
            self.rand_index = rand_score(self.class_labels, self.predictions)
        if self.adjusted_rand_index is None or overwrite:
            self.adjusted_rand_index = adjusted_rand_score(
                self.class_labels, self.predictions
            )
        if self.mutual_information is None or overwrite:
            self.mutual_information = mutual_info_score(
                self.class_labels, self.predictions
            )
        if self.adjusted_mutual_information is None or overwrite:
            self.adjusted_mutual_information = adjusted_mutual_info_score(
                self.class_labels, self.predictions
            )
        if self.normalised_mutual_information is None or overwrite:
            self.normalised_mutual_information = normalized_mutual_info_score(
                self.class_labels, self.predictions
            )

    def infer_size(self, overwrite=False):
        """
        Infer and return the size of the dataset used in the results.

        This method estimates the size of the dataset that was used for the estimator,
        based on the results data.

        Also infers the number of clusters generated.

        Parameters
        ----------
        overwrite : bool, default=False
            If the function should overwrite the current values when they are not None.
        """
        if self.n_cases is None or overwrite:
            self.n_cases = len(self.class_labels)
        if self.n_clusters is None or overwrite:
            self.n_clusters = len(self.probabilities[0])


def load_clusterer_results(file_path, calculate_stats=True, verify_values=True):
    """
    Load and return clusterer results from a specified file.

    This function reads a file containing clusterer results and reconstructs the
    ClustererResults object. It optionally calculates performance statistics and
    verifies values based on the loaded data.

    Parameters
    ----------
    file_path : str
        The path to the file from which clusterer results should be loaded. The file
        should be a tsml formatted clusterer results file.
    calculate_stats : bool, default=True
        Whether to calculate performance statistics from the loaded results.
    verify_values : bool, default=True
        If the function should perform verification of the loaded values.

    Returns
    -------
    cr : ClustererResults
        A ClustererResults object containing the results loaded from the file.
    """
    with open(file_path) as file:
        lines = file.readlines()

        line1 = lines[0].split(",")
        line3 = lines[2].split(",")
        cl_acc = float(line3[0])
        n_clusters = int(line3[6])
        n_cases = len(lines) - 3

        line_size = len(lines[3].split(","))

        class_labels = np.zeros(n_cases)
        cluster = np.zeros(n_cases)
        probabilities = np.zeros((n_cases, n_clusters))

        if line_size > 4 + n_clusters:
            pred_times = np.zeros(n_cases)
        else:
            pred_times = None

        if line_size > 6 + n_clusters:
            pred_descriptions = []
        else:
            pred_descriptions = None

        for i in range(0, n_cases):
            line = lines[i + 3].split(",")
            class_labels[i] = int(line[0])
            cluster[i] = int(line[1])

            for j in range(0, n_clusters):
                probabilities[i, j] = float(line[3 + j])

            if pred_times is not None:
                pred_times[i] = float(line[4 + n_clusters])

            if pred_descriptions is not None:
                pred_descriptions.append(",".join(line[6 + n_clusters :]).strip())

    cr = ClustererResults(
        dataset_name=line1[0],
        clusterer_name=line1[1],
        split=line1[2],
        resample_id=None if line1[3] == "None" else int(line1[3]),
        time_unit=line1[4].lower(),
        description=",".join(line1[5:]).strip(),
        parameters=lines[1].strip(),
        fit_time=float(line3[1]),
        predict_time=float(line3[2]),
        benchmark_time=float(line3[3]),
        memory_usage=float(line3[4]),
        n_classes=int(line3[5]),
        n_clusters=n_clusters,
        class_labels=class_labels,
        predictions=cluster,
        probabilities=probabilities,
        pred_times=pred_times,
        pred_descriptions=pred_descriptions,
    )

    if calculate_stats:
        cr.calculate_statistics()

    if verify_values:
        cr.infer_size(overwrite=True)
        assert cr.n_cases == n_cases
        assert cr.n_clusters == n_clusters

        assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-6)

        if calculate_stats:
            assert cr.clustering_accuracy == cl_acc

    return cr
