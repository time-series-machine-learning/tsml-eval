"""Class for storing and loading results from a classification experiment."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

import tsml_eval.evaluation.storage as storage
from tsml_eval.evaluation.storage.estimator_results import EstimatorResults
from tsml_eval.utils.experiments import write_classification_results


class ClassifierResults(EstimatorResults):
    """
    A class for storing and managing results from classification experiments.

    This class provides functionalities for storing classification results,
    including predictions, probabilities, and various performance metrics.
    It extends the `EstimatorResults` class, inheriting its base functionalities.

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset used, by default "N/A".
    classifier_name : str, optional
        Name of the classifier used, by default "N/A".
    split : str, optional
        Type of data split used, by default "N/A".
    resample_id : int or None, optional
        Identifier for the resampling method, by default None.
    time_unit : str, optional
        Unit of time measurement, by default "nanoseconds".
    description : str, optional
        Description of the classification experiment, by default "".
    parameters : str, optional
        Information about parameters used, by default "No parameter info".
    fit_time : float, optional
        Time taken for fitting the model, by default -1.0.
    predict_time : float, optional
        Time taken for making predictions, by default -1.0.
    benchmark_time : float, optional
        Time taken for benchmarking, by default -1.0.
    memory_usage : float, optional
        Memory usage during the experiment, by default -1.0.
    n_classes : int or None, optional
        Number of classes in the classification task, by default None.
    error_estimate_method : str, optional
        Method used for error estimation, by default "N/A".
    error_estimate_time : float, optional
        Time taken for error estimation, by default -1.0.
    build_plus_estimate_time : float, optional
        Total time for building and estimating, by default -1.0.
    class_labels : array-like or None, optional
        Actual class labels, by default None.
    predictions : array-like or None, optional
        Predicted class labels, by default None.
    probabilities : array-like or None, optional
        Predicted class probabilities, by default None.
    pred_times : array-like or None, optional
        Prediction times for each instance, by default None.
    pred_descriptions : list of str or None, optional
        Descriptions for each prediction, by default None.

    """

    def __init__(
        self,
        dataset_name="N/A",
        classifier_name="N/A",
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
        error_estimate_method="N/A",
        error_estimate_time=-1.0,
        build_plus_estimate_time=-1.0,
        class_labels=None,
        predictions=None,
        probabilities=None,
        pred_times=None,
        pred_descriptions=None,
    ):
        # Line 3
        self.n_classes = n_classes
        self.train_estimate_method = error_estimate_method
        self.train_estimate_time = error_estimate_time
        self.fit_and_estimate_time = build_plus_estimate_time

        # Results
        self.class_labels = class_labels
        self.predictions = predictions
        self.probabilities = probabilities
        self.pred_times = pred_times
        self.pred_descriptions = pred_descriptions

        self.n_cases = None

        self.accuracy = None
        self.balanced_accuracy = None
        self.f1_score = None
        self.negative_log_likelihood = None
        self.mean_auroc = None

        super(ClassifierResults, self).__init__(
            dataset_name=dataset_name,
            estimator_name=classifier_name,
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
        "accuracy": ("Accuracy", True),
        "balanced_accuracy": ("BalAcc", True),
        "f1_score": ("F1", True),
        "negative_log_likelihood": ("NLL", False),
        "mean_auroc": ("AUROC", True),
        **EstimatorResults.statistics,
    }

    def save_to_file(self, file_path, full_path=True):
        """
        Save the classifier results to a specified file.

        This method serializes the results of the classifier and saves them to a file
        in a chosen format.

        Parameters
        ----------
        file_path : str
            The path to the file where the results should be saved.
        """
        self.infer_size()

        if self.accuracy is None:
            self.accuracy = accuracy_score(self.class_labels, self.predictions)

        write_classification_results(
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
            accuracy=self.accuracy,
            fit_time=self.fit_time,
            predict_time=self.predict_time,
            benchmark_time=self.benchmark_time,
            memory_usage=self.memory_usage,
            n_classes=self.n_classes,
            train_estimate_method=self.train_estimate_method,
            train_estimate_time=self.train_estimate_time,
            fit_and_estimate_time=self.fit_and_estimate_time,
        )

    def load_from_file(self, file_path):
        """
        Load classifier results from a specified file.

        This method deserializes classifier results from a given file, allowing for the
        analysis and comparison of previously computed results.

        Parameters
        ----------
        file_path : str
            The path to the file from which the results should be loaded.

        Returns
        -------
        self: ClassifierResults
            The classifier results object loaded from the file.
        """
        cr = storage.load_classifier_results(file_path)
        self.__dict__.update(cr.__dict__)
        return self

    def calculate_statistics(self, overwrite=False):
        """
        Calculate and return various statistics based on the classifier results.

        This method computes various performance metrics, such as accuracy, F1 score,
        and others, based on the classifier's output.

        Returns
        -------
        dict
            A dictionary containing the calculated statistics. Keys are the names of the
            metrics, and values are their computed values.
        """
        self.infer_size(overwrite=overwrite)

        if self.accuracy is None or overwrite:
            self.accuracy = accuracy_score(self.class_labels, self.predictions)
        if self.balanced_accuracy is None or overwrite:
            self.balanced_accuracy = balanced_accuracy_score(
                self.class_labels, self.predictions
            )
        if self.f1_score is None or overwrite:
            self.f1_score = f1_score(
                self.class_labels, self.predictions, average="macro"
            )
        if self.negative_log_likelihood is None or overwrite:
            self.negative_log_likelihood = log_loss(
                self.class_labels, self.probabilities
            )
        if self.mean_auroc is None or overwrite:
            self.mean_auroc = roc_auc_score(
                self.class_labels,
                self.predictions if self.n_classes == 2 else self.probabilities,
                multi_class="ovr",
            )

    def infer_size(self, overwrite=False):
        """
        Infer and return the size of the dataset used in the classifier.

        This method estimates the size of the dataset that was used for the classifier, based on the results data.

        Returns
        -------
        int
            The inferred size of the dataset.

        Notes
        -----
        The accuracy of the inferred size may vary and should be validated with actual dataset parameters when possible.
        """
        if self.n_cases is None or overwrite:
            self.n_cases = len(self.class_labels)
        if self.n_classes is None or overwrite:
            self.n_classes = len(self.probabilities[0])


def load_classifier_results(file_path, calculate_stats=True, verify_values=True):
    """
    Load and return classifier results from a specified file.

    This function reads a file containing serialized classifier results and
    deserializes it to reconstruct the classifier results object. It optionally
    calculates statistics and verifies values based on the loaded data.

    Parameters
    ----------
    file_path : str
        The path to the file from which classifier results should be loaded. The file should be in a format compatible with the serialization method used.
    calculate_stats : bool, optional
        A flag to indicate whether to calculate statistics from the loaded results. Default is True.
    verify_values : bool, optional
        A flag to determine if the function should perform verification of the loaded values. Default is True.

    Returns
    -------
    ClassifierResults
        A ClassifierResults object containing the results loaded from the file.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

        line1 = lines[0].split(",")
        line3 = lines[2].split(",")
        acc = float(line3[0])
        n_classes = int(line3[5])
        n_cases = len(lines) - 3

        line_size = len(lines[3].split(","))

        class_labels = np.zeros(n_cases)
        predictions = np.zeros(n_cases)
        probabilities = np.zeros((n_cases, n_classes))

        if line_size > 3 + n_classes:
            pred_times = np.zeros(n_cases)
        else:
            pred_times = None

        if line_size > 6 + n_classes:
            pred_descriptions = []
        else:
            pred_descriptions = None

        for i in range(0, n_cases):
            line = lines[i + 3].split(",")
            class_labels[i] = int(line[0])
            predictions[i] = int(line[1])

            for j in range(0, n_classes):
                probabilities[i, j] = float(line[3 + j])

            if pred_times is not None:
                pred_times[i] = float(line[5 + n_classes])

            if pred_descriptions is not None:
                pred_descriptions.append(",".join(line[6 + n_classes :]).strip())

    cr = ClassifierResults(
        dataset_name=line1[0],
        classifier_name=line1[1],
        split=line1[2],
        resample_id=None if line1[3] == "None" else int(line1[3]),
        time_unit=line1[4].lower(),
        description=",".join(line1[5:]).strip(),
        parameters=lines[1].strip(),
        fit_time=float(line3[1]),
        predict_time=float(line3[2]),
        benchmark_time=float(line3[3]),
        memory_usage=float(line3[4]),
        n_classes=n_classes,
        error_estimate_method=line3[6],
        error_estimate_time=float(line3[7]),
        build_plus_estimate_time=float(line3[8]),
        class_labels=class_labels,
        predictions=predictions,
        probabilities=probabilities,
        pred_times=pred_times,
        pred_descriptions=pred_descriptions,
    )

    if calculate_stats:
        cr.calculate_statistics()

    if verify_values:
        cr.infer_size(overwrite=True)
        assert cr.n_cases == n_cases
        assert cr.n_classes == n_classes

        if calculate_stats:
            assert cr.accuracy == acc

    return cr
