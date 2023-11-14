"""Class for storing and loading results from a clustering experiment."""

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
)

import tsml_eval.evaluation.storage as storage
from tsml_eval.evaluation.metrics import clustering_accuracy_score
from tsml_eval.evaluation.storage.estimator_results import EstimatorResults
from tsml_eval.utils.experiments import write_clustering_results


class ClustererResults(EstimatorResults):
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

        super(ClustererResults, self).__init__(
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

    # var_name: (display_name, higher is better)
    statistics = {
        "clustering_accuracy": ("CLAcc", True),
        "rand_index": ("RI", True),
        "adjusted_rand_index": ("ARI", True),
        "mutual_information": ("MI", True),
        "adjusted_mutual_information": ("AMI", True),
        "normalised_mutual_information": ("NMI", True),
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

    def load_from_file(self, file_path):
        """Load results from a specified file.

        Parameters
        ----------
        file_path : str
            The path to the file where the results will be loaded from.
        """
        cr = storage.load_clusterer_results(file_path)
        self.__dict__.update(cr.__dict__)
        return self

    def calculate_statistics(self, overwrite=False):
        """Calculate statistics from the results.

        This method should handle any necessary calculations to produce statistics
        from the results data held within the object.
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
        if self.n_cases is None or overwrite:
            self.n_cases = len(self.class_labels)
        if self.n_clusters is None or overwrite:
            self.n_clusters = len(self.probabilities[0])


def load_clusterer_results(file_path, calculate_stats=True, verify_values=True):
    """Load clusterer results from a file."""

    with open(file_path, "r") as file:
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

        if line_size > 3 + n_clusters:
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
                pred_times[i] = float(line[5 + n_clusters])

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

        if calculate_stats:
            assert cr.clustering_accuracy == cl_acc

    return cr
