"""Class for storing and loading results from a classification experiment."""

import warnings

import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)

from tsml_eval.evaluation.storage.estimator_results import EstimatorResults
from tsml_eval.utils.results_writing import write_classification_results


class ClassifierResults(EstimatorResults):
    """
    A class for storing and managing results from classification experiments.

    This class provides functionalities for storing classification results,
    including predictions, probabilities, and various performance metrics.
    It extends the `EstimatorResults` class, inheriting its base functionalities.

    Parameters
    ----------
    dataset_name : str, default="N/A"
        Name of the dataset used.
    classifier_name : str, default="N/A"
        Name of the classifier used.
    split : str, default="N/A"
        Type of data split used, i.e. "train" or "test".
    resample_id : int or None, default=None
        Random seed used for the data resample, with 0 usually being the original data.
    time_unit : str, default="nanoseconds"
        Time measurement used for other fields.
    description : str, default=""
        Additional description of the classification experiment. Appended to the end
        of the first line of the results file.
    parameters : str, default="No parameter info"
        Information about parameters used in the classifier and other build information.
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
    error_estimate_method : str, default="N/A"
        Method used for train error/accuracy estimation (i.e. 10-fold CV, OOB error).
    error_estimate_time : float, default=-1.0
        Time taken for train error/accuracy estimation.
    build_plus_estimate_time : float, default=-1.0
        Total time for building the classifier and estimating error/accuracy on the
        train set. For certain methods this can be different from the sum of fit_time
        and error_estimate_time.
    class_labels : array-like or None, default=None
        Actual class labels.
    predictions : array-like or None, default=None
        Predicted class labels.
    probabilities : array-like or None, default=None
        Predicted class probabilities.
    pred_times : array-like or None, default=None
        Prediction times for each case.
    pred_descriptions : list of str or None, default=None
        Descriptions for each prediction.

    Attributes
    ----------
    n_cases : int or None
        Number of cases in the dataset.
    accuracy : float or None
        Accuracy of the classifier.
    balanced_accuracy : float or None
        Balanced accuracy of the classifier.
    auroc_score : float or None
        Mean area under the ROC curve of the classifier.
    log_loss : float or None
        Negative log likelihood of the classifier.
    sensitivity : float or None
        Sensitivity of the classifier.
    specificity : float or None
        Specificity of the classifier.
    f1_score : float or None
        F1 score of the classifier.

    Examples
    --------
    >>> from tsml_eval.evaluation.storage import ClassifierResults
    >>> from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH
    >>> cr = ClassifierResults().load_from_file(
    ...     _TEST_RESULTS_PATH +
    ...     "/classification/ROCKET/Predictions/Chinatown/testResample0.csv"
    ... )
    >>> cr.calculate_statistics()
    >>> acc = cr.accuracy
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
        # Line 1
        self.classifier_name = classifier_name

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
        self._minority_class = None
        self._majority_class = None

        self.accuracy = None
        self.balanced_accuracy = None
        self.auroc_score = None
        self.log_loss = None
        self.sensitivity = None
        self.specificity = None
        self.f1_score = None

        super().__init__(
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

    # var_name: (display_name, higher is better, is timing)
    statistics = {
        "accuracy": ("Accuracy", True, False),
        "balanced_accuracy": ("BalAcc", True, False),
        "auroc_score": ("AUROC", True, False),
        "log_loss": ("LogLoss", False, False),
        "sensitivity": ("Sensitivity", True, False),
        "specificity": ("Specificity", True, False),
        "f1_score": ("F1", True, False),
        **EstimatorResults.statistics,
    }

    def save_to_file(self, file_path, full_path=True):
        """
        Write the classifier results into a file format used by tsml.

        Parameters
        ----------
        file_path : str
            Path to write the results file to or the directory to build the default file
            structure if full_path is False.
        full_path : boolean, default=True
            If True, results are written directly to the directory passed in file_path.
            If False, then a standard file structure using the classifier and dataset
            names is created and used to write the results file.
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

    def load_from_file(self, file_path, verify_values=True):
        """
        Load classifier results from a specified file.

        This method reads a file containing classifier results and reconstructs the
        ClassifierResults object. It calculates performance statistics and
        verifies values based on the loaded data.

        Parameters
        ----------
        file_path : str
            The path to the file from which classifier results should be loaded. The
            file should be a tsml formatted classifier results file.
        verify_values : bool, default=True
            If the method should perform verification of the loaded values.

        Returns
        -------
        self : ClassifierResults
            The same ClassifierResults object with loaded results.
        """
        cr = load_classifier_results(file_path, verify_values=verify_values)
        self.__dict__.update(cr.__dict__)
        return self

    def calculate_statistics(self, overwrite=False):
        """
        Calculate various performance statistics based on the classifier results.

        This method computes various performance metrics, such as accuracy, F1 score,
        and others, based on the classifiers output.

        Parameters
        ----------
        overwrite : bool, default=False
            If the function should overwrite the current values when they are not None.
        """
        self.infer_size(overwrite=overwrite)

        if self.accuracy is None or overwrite:
            self.accuracy = accuracy_score(self.class_labels, self.predictions)
        if self.balanced_accuracy is None or overwrite:
            self.balanced_accuracy = balanced_accuracy_score(
                self.class_labels, self.predictions
            )
        if self.log_loss is None or overwrite:
            # We check this elsewhere, they are just stricter on the tolerance
            warnings.filterwarnings(
                "ignore",
                message="The y_pred values do not sum to one",
            )
            self.log_loss = log_loss(
                self.class_labels,
                self.probabilities,
            )
        if self.auroc_score is None or overwrite:
            self.auroc_score = roc_auc_score(
                self.class_labels,
                self.probabilities[:, 1] if self.n_classes == 2 else self.probabilities,
                average="weighted",
                multi_class="ovr",
            )
        if self.sensitivity is None or overwrite:
            self.sensitivity = recall_score(
                self.class_labels,
                self.predictions,
                average="binary" if self.n_classes == 2 else "weighted",
                pos_label=self._minority_class if self.n_classes == 2 else 1,
                zero_division=0.0,
            )
        if self.specificity is None or overwrite:
            self.specificity = recall_score(
                self.class_labels,
                self.predictions,
                average="binary" if self.n_classes == 2 else "weighted",
                pos_label=self._majority_class if self.n_classes == 2 else 1,
                zero_division=0.0,
            )
        if self.f1_score is None or overwrite:
            self.f1_score = f1_score(
                self.class_labels,
                self.predictions,
                average="binary" if self.n_classes == 2 else "weighted",
                pos_label=self._minority_class if self.n_classes == 2 else 1,
                zero_division=0.0,
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
            self.n_cases = len(self.class_labels)
        if self.n_classes is None or overwrite:
            self.n_classes = len(self.probabilities[0])
        if self._minority_class is None or self._majority_class is None or overwrite:
            unique, counts = np.unique(self.class_labels, return_counts=True)
            self._minority_class = unique[np.argmin(counts)]
            self._majority_class = unique[np.argmax(counts)]


def load_classifier_results(file_path, calculate_stats=True, verify_values=True):
    """
    Load and return classifier results from a specified file.

    This function reads a file containing classifier results and reconstructs the
    ClassifierResults object. It optionally calculates performance statistics and
    verifies values based on the loaded data.

    Parameters
    ----------
    file_path : str
        The path to the file from which classifier results should be loaded. The file
        should be a tsml formatted classifier results file.
    calculate_stats : bool, default=True
        Whether to calculate performance statistics from the loaded results.
    verify_values : bool, default=True
        If the function should perform verification of the loaded values.

    Returns
    -------
    cr : ClassifierResults
        A ClassifierResults object containing the results loaded from the file.
    """
    with open(file_path) as file:
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

        if line_size > 4 + n_classes:
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
                pred_times[i] = float(line[4 + n_classes])

            if pred_descriptions is not None:
                pred_descriptions.append(",".join(line[6 + n_classes :]).strip())

        # compatability with old results files
        if len(line3) > 6:
            error_estimate_method = line3[6]
            error_estimate_time = float(line3[7])
            build_plus_estimate_time = float(line3[8])
        else:
            error_estimate_method = "N/A"
            error_estimate_time = -1.0
            build_plus_estimate_time = -1.0

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
        error_estimate_method=error_estimate_method,
        error_estimate_time=error_estimate_time,
        build_plus_estimate_time=build_plus_estimate_time,
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

        assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-5)

        if calculate_stats:
            assert cr.accuracy == acc

    return cr
