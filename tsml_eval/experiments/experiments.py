"""Functions to perform machine learning/data mining experiments.

Results are saved a standardised format used by tsml.
"""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = [
    "run_classification_experiment",
    "load_and_run_classification_experiment",
    "run_regression_experiment",
    "load_and_run_regression_experiment",
    "run_classification_experiment",
    "load_and_run_clustering_experiment",
    "run_forecasting_experiment",
    "load_and_run_forecasting_experiment",
]

import os
import tempfile
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from aeon.benchmarking.metrics.clustering import clustering_accuracy_score
from aeon.classification import BaseClassifier
from aeon.clustering import BaseClusterer
from aeon.forecasting import BaseForecaster, RegressionForecaster
from aeon.regression.base import BaseRegressor
from aeon.transformations.series import TrainTestTransformer
from aeon.utils.validation import get_n_cases
from sklearn import preprocessing
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import cross_val_predict
from tsml.base import BaseTimeSeriesEstimator
from tsml.compose import (
    SklearnToTsmlClassifier,
    SklearnToTsmlClusterer,
    SklearnToTsmlRegressor,
)
from tsml.utils.validation import is_clusterer

from tsml_eval.utils.datasets import load_experiment_data
from tsml_eval.utils.experiments import (
    _check_existing_results,
    estimator_attributes_to_file,
    timing_benchmark,
)
from tsml_eval.utils.functions import time_function
from tsml_eval.utils.memory_recorder import record_max_memory
from tsml_eval.utils.resampling import resample_data, stratified_resample_data
from tsml_eval.utils.results_writing import (
    regression_results_third_line,
    results_third_line,
    write_classification_results,
    write_clustering_results,
    write_results_to_tsml_format,
)

MEMRECORD_ENV = os.getenv("MEMRECORD_INTERVAL")
if isinstance(MEMRECORD_ENV, str):  # pragma: no cover
    MEMRECORD_INTERVAL = float(MEMRECORD_ENV)
else:
    MEMRECORD_INTERVAL = 5.0


def run_classification_experiment(
    X_train: np.ndarray | list,
    y_train: np.ndarray,
    X_test: np.ndarray | list,
    y_test: np.ndarray,
    classifier,
    results_path,
    classifier_name=None,
    dataset_name="N/A",
    resample_id=None,
    data_transforms=None,
    transform_train_only=False,
    build_test_file=True,
    build_train_file=False,
    ignore_custom_train_estimate=False,
    attribute_file_path=None,
    att_max_shape=0,
    benchmark_time=True,
):
    """Run a classification experiment and save the results to file.

    Function to run a basic classification experiment for a
    <dataset>/<classifier>/<resample> combination and write the results to csv file(s)
    at a given location.

    Parameters
    ----------
    X_train : np.ndarray or list of np.ndarray
        The data to train the classifier. Numpy array or list of numpy arrays in the
        ``aeon`` data format.
    y_train : np.array
        Training data class labels. One label per case in the training data using the
        same ordering.
    X_test : np.ndarray or list of np.ndarray
        The data used to test the trained classifier. Numpy array or list of numpy
        arrays in the ``aeon`` data format.
    y_test : np.array
        Testing data class labels. One label per case in the testing data using the
        same ordering.
    classifier : BaseClassifier
        Classifier to be used in the experiment.
    results_path : str
        Location of where to write results. Any required directories will be created.
    classifier_name : str or None, default=None
        Name of classifier used in writing results. If None, the name is taken from
        the classifier.
    dataset_name : str, default="N/A"
        Name of dataset.
    resample_id : int or None, default=None
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    data_transforms : transformer, list of transformers or None, default=None
        Transformer(s) to apply to the data before running the experiment.
        If a list, the transformers are applied in order.
        If None, no transformation is applied.
        Calls fit_transform on the training data and transform on the test data.
    transform_train_only : bool, default=False
        if True, the data_transforms are limited to the training data only.
    build_test_file : bool, default=True:
        Whether to generate test files or not. If the classifier can generate its own
        train probabilities, the classifier will be built but no file will be output.
    build_train_file : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the classifier can produce its
        own estimates, those are used instead.
    ignore_custom_train_estimate : bool, default=False
        todo
    attribute_file_path : str or None, default=None
        todo (only test)
    att_max_shape : int, default=0
        todo
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    """
    if not build_test_file and not build_train_file:
        raise ValueError(
            "Both test_file and train_file are set to False. "
            "At least one must be written."
        )

    if classifier_name is None:
        classifier_name = type(classifier).__name__

    use_fit_predict = False
    if isinstance(classifier, BaseClassifier):
        if not ignore_custom_train_estimate and classifier.get_tag(
            "capability:train_estimate", False, False
        ):
            use_fit_predict = True
    elif isinstance(classifier, BaseTimeSeriesEstimator) and is_classifier(classifier):
        pass
    elif isinstance(classifier, BaseEstimator) and is_classifier(classifier):
        classifier = SklearnToTsmlClassifier(
            classifier=classifier,
            pad_unequal=True,
            concatenate_channels=True,
            clone_estimator=False,
            random_state=(
                classifier.random_state if hasattr(classifier, "random_state") else None
            ),
        )
    else:
        raise TypeError("classifier must be a tsml, aeon or sklearn classifier.")

    n_cases_test = get_n_cases(X_test)
    if data_transforms is not None:
        if not isinstance(data_transforms, list):
            data_transforms = [data_transforms]

        for transform in data_transforms:
            transform_results = transform.fit_transform(X_train, y_train)
            if isinstance(transform_results, tuple) and len(transform_results) == 2:
                # If the transformer returns a tuple of length 2, assume it is (X, y)
                X_train, y_train = transform_results
            else:
                X_train = transform_results

            if not transform_train_only:
                transform_results = transform.transform(X_test, y_test)
                if isinstance(transform_results, tuple) and len(transform_results) == 2:
                    X_test, y_test = transform_results
                else:
                    X_test = transform_results

                # If we have edited the number of cases in test something has gone
                # wrong i.e. we have applied SMOTE to the test set
                new_n_cases_test = get_n_cases(X_test)
                assert new_n_cases_test == n_cases_test, (
                    f"Error: X_test sample size changed from {n_cases_test} to "
                    f"{new_n_cases_test} after transformation "
                    f"{transform.__class__.__name__}"
                )

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    encoder_dict = {label: i for i, label in enumerate(le.classes_)}
    n_classes = len(np.unique(y_train))

    needs_fit = True
    fit_time = -1
    mem_usage = -1
    benchmark = -1
    train_time = -1
    fit_and_train_time = -1

    if benchmark_time:
        benchmark = timing_benchmark(random_state=resample_id)

    first_comment = (
        "Generated by run_classification_experiment on "
        f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}. "
        f"Encoder dictionary: {str(encoder_dict)}. "
        f"Data transformers: {str(data_transforms)}. "
    )

    second = str(classifier.get_params()).replace("\n", " ").replace("\r", " ")

    if build_train_file:
        cv_size = 10
        start = int(round(time.time() * 1000))
        if use_fit_predict:
            train_probs = classifier.fit_predict_proba(X_train, y_train)
            needs_fit = False
            fit_and_train_time = int(round(time.time() * 1000)) - start
        else:
            _, counts = np.unique(y_train, return_counts=True)
            min_class = max(2, np.min(counts))
            if min_class < cv_size:
                cv_size = min_class

            train_probs = cross_val_predict(
                classifier, X_train, y=y_train, cv=cv_size, method="predict_proba"
            )
            train_time = int(round(time.time() * 1000)) - start

        train_preds = np.unique(y_train)[np.argmax(train_probs, axis=1)]
        train_acc = accuracy_score(y_train, train_preds)

        write_classification_results(
            train_preds,
            train_probs,
            y_train,
            classifier_name,
            dataset_name,
            results_path,
            full_path=False,
            first_line_classifier_name=(
                f"{classifier_name} ({type(classifier).__name__})"
            ),
            split="TRAIN",
            resample_id=resample_id,
            time_unit="MILLISECONDS",
            first_line_comment=first_comment,
            parameter_info=second,
            accuracy=train_acc,
            fit_time=fit_time,
            predict_time=-1,
            benchmark_time=benchmark,
            memory_usage=mem_usage,
            n_classes=n_classes,
            train_estimate_method="Custom" if use_fit_predict else f"{cv_size}F-CV",
            train_estimate_time=train_time,
            fit_and_estimate_time=fit_and_train_time,
        )

    if build_test_file:
        if needs_fit:
            mem_usage, fit_time = record_max_memory(
                classifier.fit,
                args=(X_train, y_train),
                interval=MEMRECORD_INTERVAL,
                return_func_time=True,
            )
            fit_time += int(round(getattr(classifier, "_fit_time_milli", 0)))

        if attribute_file_path is not None:
            estimator_attributes_to_file(
                classifier, attribute_file_path, max_list_shape=att_max_shape
            )

        start = int(round(time.time() * 1000))
        test_probs = classifier.predict_proba(X_test)
        test_time = (
            int(round(time.time() * 1000))
            - start
            + int(round(getattr(classifier, "_predict_time_milli", 0)))
        )

        test_preds = classifier.classes_[np.argmax(test_probs, axis=1)]
        test_acc = accuracy_score(y_test, test_preds)

        write_classification_results(
            test_preds,
            test_probs,
            y_test,
            classifier_name,
            dataset_name,
            results_path,
            full_path=False,
            first_line_classifier_name=(
                f"{classifier_name} ({type(classifier).__name__})"
            ),
            split="TEST",
            resample_id=resample_id,
            time_unit="MILLISECONDS",
            first_line_comment=first_comment,
            parameter_info=second,
            accuracy=test_acc,
            fit_time=fit_time,
            predict_time=test_time,
            benchmark_time=benchmark,
            memory_usage=mem_usage,
            n_classes=n_classes,
            train_estimate_method="N/A",
            train_estimate_time=-1,
            fit_and_estimate_time=fit_and_train_time,
        )


def load_and_run_classification_experiment(
    problem_path,
    results_path,
    dataset,
    classifier,
    classifier_name=None,
    resample_id=0,
    data_transforms=None,
    transform_train_only=False,
    build_train_file=False,
    write_attributes=False,
    att_max_shape=0,
    benchmark_time=True,
    overwrite=False,
    predefined_resample=False,
):
    """Load a dataset and run a classification experiment.

    Function to load a dataset, run a basic classification experiment for a
    <dataset>/<classifier>/<resample> combination, and write the results to csv file(s)
    at a given location.

    Parameters
    ----------
    problem_path : str
        Location of problem files, full path.
    results_path : str
        Location of where to write results. Any required directories will be created.
    dataset : str
        Name of problem. Files must be <problem_path>/<dataset>/<dataset>+"_TRAIN.ts",
        same for "_TEST.ts".
    classifier : BaseClassifier
        Classifier to be used in the experiment.
    classifier_name : str or None, default=None
        Name of classifier used in writing results. If None, the name is taken from
        the classifier.
    resample_id : int, default=0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    data_transforms : transformer, list of transformers or None, default=None
        Transformer(s) to apply to the data before running the experiment.
        If a list, the transformers are applied in order.
        If None, no transformation is applied.
        Calls fit_transform on the training data and transform on the test data.
    transform_train_only : bool, default=False
        if the data_transforms are limited to the training data only.
    build_train_file : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the classifier can produce its
        own estimates, those are used instead.
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    overwrite : bool, default=False
        If set to False, this will only build results if there is not a result file
        already present. If True, it will overwrite anything already there.
    predefined_resample : bool, default=False
        Read a predefined resample from file instead of performing a resample. If True
        the file format must include the resample_id at the end of the dataset name i.e.
        <problem_path>/<dataset>/<dataset>+<resample_id>+"_TRAIN.ts".
    """
    if classifier_name is None:
        classifier_name = type(classifier).__name__

    build_test_file, build_train_file = _check_existing_results(
        results_path,
        classifier_name,
        dataset,
        resample_id,
        overwrite,
        True,
        build_train_file,
    )

    if not build_test_file and not build_train_file:
        warnings.warn("All files exist and not overwriting, skipping.", stacklevel=1)
        return

    X_train, y_train, X_test, y_test, resample = load_experiment_data(
        problem_path, dataset, resample_id, predefined_resample
    )

    if resample:
        X_train, y_train, X_test, y_test = stratified_resample_data(
            X_train, y_train, X_test, y_test, random_state=resample_id
        )

    if write_attributes:
        attribute_file_path = f"{results_path}/{classifier_name}/Workspace/{dataset}/"
    else:
        attribute_file_path = None

    run_classification_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier,
        results_path,
        classifier_name=classifier_name,
        dataset_name=dataset,
        resample_id=resample_id,
        data_transforms=data_transforms,
        transform_train_only=transform_train_only,
        build_test_file=build_test_file,
        build_train_file=build_train_file,
        attribute_file_path=attribute_file_path,
        att_max_shape=att_max_shape,
        benchmark_time=benchmark_time,
    )


def transform_input(
    data_transforms,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray = None,
    y_test: np.ndarray = None,
):
    if data_transforms is not None:
        if not isinstance(data_transforms, list):
            data_transforms = [data_transforms]

        for transform in data_transforms:
            x_train = transform.fit_transform(x_train, y_train)
            x_test = transform.transform(x_test, y_test)
    return x_train, x_test


def cross_validate_train_data(estimator, y_train, X_train):
    cv_size = min(10, len(y_train))
    start = int(round(time.time() * 1000))
    train_preds = cross_val_predict(estimator, X_train, y=y_train, cv=cv_size)
    train_time = int(round(time.time() * 1000)) - start
    train_estimate_method = f"{cv_size}F-CV"
    return train_preds, train_time, train_estimate_method


class Experiment:
    """Run an experiment and save the results to file.

    Function to run a basic experiment for a
    <dataset>/<estimator>/<resample> combination and write the results to csv file(s)
    at a given location.

    Parameters
    ----------
    X_train : np.ndarray or list of np.ndarray
        The data to train the classifier. Numpy array or list of numpy arrays in the
        ``aeon`` data format.
    y_train : np.array
        Training data class labels. One label per case in the training data using the
        same ordering.
    X_test : np.ndarray or list of np.ndarray
        The data used to test the trained classifier. Numpy array or list of numpy
        arrays in the ``aeon`` data format.
    y_test : np.array
        Testing data class labels. One label per case in the testing data using the
        same ordering.
    estimator : BaseRegressor
        Estimator to be used in the experiment.
    results_path : str
        Location of where to write results. Any required directories will be created.
    estimator_name : str or None, default=None
        Name of estimator used in writing results. If None, the name is taken from
        the estimator.
    dataset_name : str, default="N/A"
        Name of dataset.
    resample_id : int or None, default=None
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    data_transforms : transformer, list of transformers or None, default=None
        Transformer(s) to apply to the data before running the experiment.
        If a list, the transformers are applied in order.
        If None, no transformation is applied.
        Calls fit_transform on the training data and transform on the test data.
    build_test_file : bool, default=True:
        Whether to generate test files or not. If the estimator can generate its own
        train predictions, the classifier will be built but no file will be output.
    build_train_file : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the estimator can produce its
        own estimates, those are used instead.
    ignore_custom_train_estimate : bool, default=False
        todo
    attribute_file_path : str or None, default=None
        todo (only test)
    att_max_shape : int, default=0
        todo
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    """

    def __init__(
        self,
        estimator,
        results_path,
        estimator_name=None,
        dataset_name="",
        resample_id=None,
        data_transforms=None,
        overwrite=False,
        build_train_file=False,
        write_attributes=False,
        att_max_shape=0,
        benchmark_time=True,
    ):
        build_test_file, build_train_file = _check_existing_results(
            results_path,
            estimator_name,
            dataset_name,
            resample_id,
            overwrite,
            True,
            build_train_file,
        )

        if not build_test_file and not build_train_file:
            warnings.warn(
                "All files exist and not overwriting, skipping.", stacklevel=1
            )
            return None

        if write_attributes:
            attribute_file_path = (
                f"{results_path}/{estimator_name}/Workspace/{dataset_name}/"
            )
        else:
            attribute_file_path = None

        # Ensure labels are floats
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        self.build_train_file = build_train_file
        self.build_test_file = build_test_file
        self.data_transforms = data_transforms
        self.benchmark_time = benchmark_time
        self.results_path = results_path
        self.dataset_name = dataset_name
        self.resample_id = resample_id
        self.benchmark = -1
        if estimator_name is None:
            self.estimator_name = type(estimator).__name__
        else:
            self.estimator_name = estimator_name
        self.estimator = self.validate_estimator(estimator=estimator)
        self.second_comment = (
            str(estimator.get_params()).replace("\n", " ").replace("\r", " ")
        )
        if attribute_file_path is not None:
            estimator_attributes_to_file(
                self.estimator, attribute_file_path, max_list_shape=att_max_shape
            )

    def run_experiment(self):
        x_train, y_train, x_test, y_test = self.load_experimental_data()

        self.first_comment = (
            "Generated by run_experiment on "
            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}"
        )

        x_train, x_test = transform_input(
            data_transforms=self.data_transforms,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
        if self.benchmark_time:
            self.benchmark = timing_benchmark(random_state=self.resample_id)

        if self.build_train_file:
            train_preds, train_time = self.generate_train_preds(x_train, y_train)
            self.write_results(
                "TRAIN", y_train, train_preds, train_time, -1, self.benchmark, -1
            )

        if self.build_test_file:
            if self.needs_fit():
                mem_usage, fit_time = record_max_memory(
                    self.estimator.fit,
                    args=(x_train, y_train),
                    interval=MEMRECORD_INTERVAL,
                    return_func_time=True,
                )
                fit_time += int(round(getattr(self.estimator, "_fit_time_milli", 0)))
            test_preds, test_time = self.generate_test_preds(x_test, y_test)
            test_time += int(round(getattr(self.estimator, "_predict_time_milli", 0)))
            self.write_results(
                "TEST",
                y_test,
                test_preds,
                fit_time,
                test_time,
                self.benchmark,
                mem_usage,
            )

    def load_experimental_data(self):
        return None, None, None, None

    def validate_estimator(self, estimator):
        estimator

    def generate_train_preds(self, X_train, y_train):
        return time_function(self.estimator.fit_predict, (X_train, y_train))

    def generate_test_preds(self, x_test, y_test):
        return time_function(self.estimator.predict, x_test)

    def needs_fit(self):
        return False

    def write_results(
        self, split, y, preds, fit_time, predict_time, benchmark_time, memory_usage
    ):
        third_line = self.get_third_line(
            y, preds, fit_time, predict_time, benchmark_time, memory_usage
        )
        write_results_to_tsml_format(
            preds,
            y,
            self.estimator_name,
            self.dataset_name,
            self.results_path,
            full_path=False,
            first_line_estimator_name=f"{self.estimator_name} ({type(self.estimator).__name__})",
            split=split,
            resample_id=self.resample_id,
            time_unit="MILLISECONDS",
            first_line_comment=self.first_comment,
            second_line=self.second_comment,
            third_line=third_line,
        )

    def get_third_line(
        self, y, preds, fit_time, predict_time, benchmark_time, memory_usage
    ):
        return results_third_line(
            y=y,
            preds=preds,
            fit_time=fit_time,
            predict_time=predict_time,
            benchmark_time=benchmark_time,
            memory_usage=memory_usage,
        )


class ForecastingExperiment(Experiment):
    def __init__(self):
        pass

    def load_experimental_data(self):
        train = pd.read_csv(
            f"{self.problem_path}/{self.dataset_name}/{self.dataset_name}_TRAIN.csv",
            index_col=0,
        ).squeeze("columns")
        train = train.astype(float).to_numpy()
        test = pd.read_csv(
            f"{self.problem_path}/{self.dataset_name}/{self.dataset_name}_TEST.csv",
            index_col=0,
        ).squeeze("columns")
        test = test.astype(float).to_numpy()
        return train, None, test, None

    def generate_test_preds(self, x_test, y_test):
        # TODO Implement this and train_preds properly
        # Remove last value as we have no actual data for it
        test_preds, test_time = time_function(self.estimator.predict, x_test)
        test_preds = test_preds.flatten()[:-1]
        return test_preds, test_time

    def validate_estimator(self, estimator):
        return validate_forecaster(estimator)


class RegressionExperiment(Experiment):
    def __init__(
        self,
        ignore_custom_train_estimate=False,
        predefined_resample=False,
        problem_path="",
    ):
        self.is_fitted = False
        self.ignore_custom_train_estimate = ignore_custom_train_estimate
        self.problem_path = problem_path
        self.predefined_resample = predefined_resample

    def load_experimental_data(self):
        X_train, y_train, X_test, y_test, resample = load_experiment_data(
            self.problem_path,
            self.dataset_name,
            self.resample_id,
            self.predefined_resample,
        )

        if resample:
            X_train, y_train, X_test, y_test = resample_data(
                X_train, y_train, X_test, y_test, random_state=self.resample_id
            )
        return X_train, y_train, X_test, y_test

    def generate_train_preds(self, X_train, y_train):
        if self.estimate_train_data and not self.ignore_custom_train_estimate:
            self.train_estimate_method = "Custom"
            train_preds, train_time = time_function(
                self.estimator.fit_predict, (X_train, y_train)
            )
            self.is_fitted = True
        else:
            train_preds, train_time, self.train_estimate_method = (
                cross_validate_train_data(self.estimator, y_train, X_train)
            )
        return train_preds, train_time

    def needs_fit(self):
        return not self.is_fitted

    def get_third_line(
        self, y, preds, fit_time, predict_time, benchmark_time, memory_usage
    ):
        return regression_results_third_line(
            y=y,
            preds=preds,
            fit_time=fit_time,
            predict_time=predict_time,
            benchmark_time=benchmark_time,
            memory_usage=memory_usage,
            train_estimate_method=self.train_estimate_method,
        )

    def validate_estimator(self, estimator):
        estimator, estimate_train_data = validate_regressor(estimator)
        self.estimate_train_data = estimate_train_data
        return estimator


def validate_forecaster(estimator):
    if isinstance(estimator, BaseForecaster):
        return estimator
    else:
        try:
            estimator, _ = validate_regressor(estimator)
            return RegressionForecaster(regressor=estimator)
        except TypeError:
            raise TypeError(
                "forecaster must be an aeon forecaster or a tsml, aeon or sklearn regressor."
            )


def validate_regressor(estimator):
    estimate_train_data = False
    if isinstance(estimator, BaseRegressor):
        if estimator.get_tag("capability:train_estimate", False, False):
            estimate_train_data = True
        return estimator, estimate_train_data
    elif isinstance(estimator, BaseTimeSeriesEstimator) and is_regressor(estimator):
        return estimator, estimate_train_data
    elif isinstance(estimator, BaseEstimator) and is_regressor(estimator):
        return (
            SklearnToTsmlRegressor(
                regressor=estimator,
                pad_unequal=True,
                concatenate_channels=True,
                clone_estimator=False,
                random_state=(
                    estimator.random_state
                    if hasattr(estimator, "random_state")
                    else None
                ),
            ),
            estimate_train_data,
        )
    else:
        raise TypeError("regressor must be a tsml, aeon or sklearn regressor.")


def load_and_run_regression_experiment(
    problem_path,
    results_path,
    dataset,
    estimator,
    estimator_name=None,
    resample_id=0,
    data_transforms=None,
    build_train_file=False,
    write_attributes=False,
    att_max_shape=0,
    benchmark_time=True,
    overwrite=False,
    predefined_resample=False,
):
    """Load a dataset and run an experiment.

    Function to load a dataset, run a basic experiment for a
    <dataset>/<estimator>/<resample> combination, and write the results to csv file(s)
    at a given location.

    Parameters
    ----------
    problem_path : str
        Location of problem files, full path.
    results_path : str
        Location of where to write results. Any required directories will be created.
    dataset : str
        Name of problem. Files must be <problem_path>/<dataset>/<dataset>+"_TRAIN.ts",
        same for "_TEST.ts".
    estimator : BaseRegressor
        Estimator to be used in the experiment.
    estimator_name : str or None, default=None
        Name of estimator used in writing results. If None, the name is taken from
        the estimator.
    resample_id : int, default=0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    data_transforms : transformer, list of transformers or None, default=None
        Transformer(s) to apply to the data before running the experiment.
        If a list, the transformers are applied in order.
        If None, no transformation is applied.
        Calls fit_transform on the training data and transform on the test data.
    build_train_file : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the estimator can produce its
        own estimates, those are used instead.
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    overwrite : bool, default=False
        If set to False, this will only build results if there is not a result file
        already present. If True, it will overwrite anything already there.
    predefined_resample : bool, default=False
        Read a predefined resample from file instead of performing a resample. If True
        the file format must include the resample_id at the end of the dataset name i.e.
        <problem_path>/<dataset>/<dataset>+<resample_id>+"_TRAIN.ts".
    """
    pass


def run_clustering_experiment(
    X_train: np.ndarray | list,
    y_train: np.ndarray,
    clusterer,
    results_path,
    X_test: np.ndarray | list | None = None,
    y_test: np.ndarray | None = None,
    n_clusters=None,
    clusterer_name=None,
    dataset_name="N/A",
    resample_id=None,
    data_transforms=None,
    build_test_file=False,
    build_train_file=True,
    attribute_file_path=None,
    att_max_shape=0,
    benchmark_time=True,
):
    """Run a clustering experiment and save the results to file.

    Function to run a basic clustering experiment for a
    <dataset>/<clusterer>/<resample> combination and write the results to csv file(s)
    at a given location.

    Parameters
    ----------
    X_train : np.ndarray or list of np.ndarray
        The data to train the classifier. Numpy array or list of numpy arrays in the
        ``aeon`` data format.
    y_train : np.array
        Training data class labels. One label per case in the training data using the
        same ordering.
    clusterer : BaseClusterer
        Clusterer to be used in the experiment.
    results_path : str
        Location of where to write results. Any required directories will be created.
    X_test : np.ndarray or list of np.ndarray
        The data used to test the trained classifier. Numpy array or list of numpy
        arrays in the ``aeon`` data format.
    y_test : np.array
        Testing data class labels. One label per case in the testing data using the
        same ordering.
    n_clusters : int or None, default=None
        Number of clusters to use if the clusterer has an `n_clusters` parameter.
        If None, the clusterers default is used. If -1, the number of classes in the
        dataset is used.

        The `n_clusters` parameter for arguments which are estimators will also be
        set to this value if it exists. Please ensure that the argument input itself
        has the `n_clusters` parameters and is not a default such as None. This is
        likely to be the case for parameters such as `estimator` or `clusterer` in
        pipelines and deep learners.
    clusterer_name : str or None, default=None
        Name of clusterer used in writing results. If None, the name is taken from
        the clusterer.
    dataset_name : str, default="N/A"
        Name of dataset.
    resample_id : int or None, default=None
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    data_transforms : transformer, list of transformers or None, default=None
        Transformer(s) to apply to the data before running the experiment.
        If a list, the transformers are applied in order.
        If None, no transformation is applied.
        Calls fit_transform on the training data and transform on the test data.
    build_test_file : bool, default=False:
        Whether to generate test files or not. If True, X_test and y_test must be
        provided.
    build_train_file : bool, default=True
        Whether to generate train files or not. The clusterer is fit using train data
        regardless of input.
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    """
    if not build_test_file and not build_train_file:
        raise ValueError(
            "Both test_file and train_file are set to False. "
            "At least one must be written."
        )

    if clusterer_name is None:
        clusterer_name = type(clusterer).__name__

    if isinstance(clusterer, BaseClusterer) or (
        isinstance(clusterer, BaseTimeSeriesEstimator) and is_clusterer(clusterer)
    ):
        pass
    elif isinstance(clusterer, BaseEstimator) and is_clusterer(clusterer):
        clusterer = SklearnToTsmlClusterer(
            clusterer=clusterer,
            pad_unequal=True,
            concatenate_channels=True,
            clone_estimator=False,
            random_state=(
                clusterer.random_state if hasattr(clusterer, "random_state") else None
            ),
        )
    else:
        raise TypeError("clusterer must be a tsml, aeon or sklearn clusterer.")

    if build_test_file and (X_test is None or y_test is None):
        raise ValueError("Test data and labels not provided, cannot build test file.")

    if data_transforms is not None:
        if not isinstance(data_transforms, list):
            data_transforms = [data_transforms]

        for transform in data_transforms:
            X_train = transform.fit_transform(X_train, y_train)
            if build_test_file:
                X_test = transform.transform(X_test, y_test)

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    if build_test_file:
        y_test = le.transform(y_test)

    encoder_dict = {label: i for i, label in enumerate(le.classes_)}
    n_classes = len(np.unique(y_train))

    benchmark = -1
    if benchmark_time:
        benchmark = timing_benchmark(random_state=resample_id)

    first_comment = (
        "Generated by run_clustering_experiment on "
        f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}. "
        f"Encoder dictionary: {str(encoder_dict)}"
    )

    # set n_clusters for clusterer and any contained estimators
    # NOTE: If the clusterer has an estimator parameteri.e. `estimator` or `clusterer`
    # which defaults to None, we cannot set the n_clusters parameter for it here.
    if isinstance(n_clusters, int):
        if n_clusters == -1:
            n_clusters = n_classes

        if "n_clusters" in clusterer.get_params():
            clusterer.set_params(n_clusters=n_clusters)
        for att in clusterer.__dict__.values():
            if isinstance(att, BaseEstimator) and "n_clusters" in att.get_params():
                att.set_params(n_clusters=n_clusters)
    elif n_clusters is not None:
        raise ValueError("n_clusters must be an int or None.")

    second = str(clusterer.get_params()).replace("\n", " ").replace("\r", " ")

    mem_usage, fit_time = record_max_memory(
        clusterer.fit,
        args=(X_train,),
        interval=MEMRECORD_INTERVAL,
        return_func_time=True,
    )
    fit_time += int(round(getattr(clusterer, "_fit_time_milli", 0)))

    if attribute_file_path is not None:
        estimator_attributes_to_file(
            clusterer, attribute_file_path, max_list_shape=att_max_shape
        )

    start = int(round(time.time() * 1000))
    if callable(getattr(clusterer, "predict_proba", None)):
        train_probs = clusterer.predict_proba(X_train)
        train_preds = np.argmax(train_probs, axis=1)
    else:
        train_preds = (
            clusterer.labels_
            if hasattr(clusterer, "labels_")
            else clusterer.predict(X_train)
        )
        train_probs = np.zeros(
            (
                len(train_preds),
                len(np.unique(train_preds)),
            )
        )
        train_probs[np.arange(len(train_preds)), train_preds] = 1
    train_time = int(round(time.time() * 1000)) - start

    if build_train_file:
        train_acc = clustering_accuracy_score(y_train, train_preds)

        write_clustering_results(
            train_preds,
            train_probs,
            y_train,
            clusterer_name,
            dataset_name,
            results_path,
            full_path=False,
            first_line_clusterer_name=f"{clusterer_name} ({type(clusterer).__name__})",
            split="TRAIN",
            resample_id=resample_id,
            time_unit="MILLISECONDS",
            first_line_comment=first_comment,
            parameter_info=second,
            clustering_accuracy=train_acc,
            fit_time=fit_time,
            predict_time=train_time,
            benchmark_time=benchmark,
            memory_usage=mem_usage,
            n_classes=n_classes,
            n_clusters=len(train_probs[0]),
        )

    if build_test_file:
        start = int(round(time.time() * 1000))
        if callable(getattr(clusterer, "predict_proba", None)):
            test_probs = clusterer.predict_proba(X_test)
            test_preds = np.argmax(test_probs, axis=1)
        else:
            test_preds = clusterer.predict(X_test)
            test_probs = np.zeros(
                (
                    len(test_preds),
                    len(np.unique(train_preds)),
                )
            )
            test_probs[np.arange(len(test_preds)), test_preds] = 1
        test_time = (
            int(round(time.time() * 1000))
            - start
            + int(round(getattr(clusterer, "_predict_time_milli", 0)))
        )

        test_acc = clustering_accuracy_score(y_test, test_preds)

        write_clustering_results(
            test_preds,
            test_probs,
            y_test,
            clusterer_name,
            dataset_name,
            results_path,
            full_path=False,
            first_line_clusterer_name=f"{clusterer_name} ({type(clusterer).__name__})",
            split="TEST",
            resample_id=resample_id,
            time_unit="MILLISECONDS",
            first_line_comment=first_comment,
            parameter_info=second,
            clustering_accuracy=test_acc,
            fit_time=fit_time,
            predict_time=test_time,
            benchmark_time=benchmark,
            memory_usage=mem_usage,
            n_classes=n_classes,
            n_clusters=len(test_probs[0]),
        )


def load_and_run_clustering_experiment(
    problem_path,
    results_path,
    dataset,
    clusterer,
    n_clusters=None,
    clusterer_name=None,
    resample_id=0,
    data_transforms=None,
    build_test_file=False,
    write_attributes=False,
    att_max_shape=0,
    benchmark_time=True,
    overwrite=False,
    predefined_resample=False,
    combine_train_test_split=False,
):
    """Load a dataset and run a clustering experiment.

    Function to load a dataset, run a basic clustering experiment for a
    <dataset>/<clusterer>/<resample> combination, and write the results to csv file(s)
    at a given location.

    Parameters
    ----------
    problem_path : str
        Location of problem files, full path.
    results_path : str
        Location of where to write results. Any required directories will be created.
    dataset : str
        Name of problem. Files must be <problem_path>/<dataset>/<dataset>+"_TRAIN.ts",
        same for "_TEST.ts".
    clusterer : BaseClusterer
        Clusterer to be used in the experiment.
    n_clusters : int or None, default=None
        Number of clusters to use if the clusterer has an `n_clusters` parameter.
        If None, the clusterers default is used. If -1, the number of classes in the
        dataset is used.

        The `n_clusters` parameter for attributes which are estimators will also be
        set to this value if it exists.
    clusterer_name : str or None, default=None
        Name of clusterer used in writing results. If None, the name is taken from
        the clusterer.
    resample_id : int, default=0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    data_transforms : transformer, list of transformers or None, default=None
        Transformer(s) to apply to the data before running the experiment.
        If a list, the transformers are applied in order.
        If None, no transformation is applied.
        Calls fit_transform on the training data and transform on the test data.
    build_test_file : bool, default=False
        Whether to generate test files or not. If true, the clusterer will assign
        clusters to the loaded test data.
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    overwrite : bool, default=False
        If set to False, this will only build results if there is not a result file
        already present. If True, it will overwrite anything already there.
    predefined_resample : bool, default=False
        Read a predefined resample from file instead of performing a resample. If True
        the file format must include the resample_id at the end of the dataset name i.e.
        <problem_path>/<dataset>/<dataset>+<resample_id>+"_TRAIN.ts".
    combine_train_test_split: bool, default=False
        Whether the train/test split should be combined. If True then
        the train/test split is combined into a single train set. If False then the
        train/test split is used as normal.
    """
    if clusterer_name is None:
        clusterer_name = type(clusterer).__name__

    if combine_train_test_split:
        build_test_file = False

    build_test_file, build_train_file = _check_existing_results(
        results_path,
        clusterer_name,
        dataset,
        resample_id,
        overwrite,
        build_test_file,
        True,
    )

    if not build_test_file and not build_train_file:
        warnings.warn("All files exist and not overwriting, skipping.", stacklevel=1)
        return

    X_train, y_train, X_test, y_test, resample = load_experiment_data(
        problem_path, dataset, resample_id, predefined_resample
    )

    if resample:
        X_train, y_train, X_test, y_test = stratified_resample_data(
            X_train, y_train, X_test, y_test, random_state=resample_id
        )

    if write_attributes:
        attribute_file_path = f"{results_path}/{clusterer_name}/Workspace/{dataset}/"
    else:
        attribute_file_path = None

    if combine_train_test_split:
        y_train = np.concatenate((y_train, y_test), axis=None)
        X_train = (
            np.concatenate([X_train, X_test], axis=0)
            if isinstance(X_train, np.ndarray)
            else X_train + X_test
        )
        X_test = None
        y_test = None

    run_clustering_experiment(
        X_train,
        y_train,
        clusterer,
        results_path,
        X_test=X_test,
        y_test=y_test,
        n_clusters=n_clusters,
        clusterer_name=clusterer_name,
        dataset_name=dataset,
        resample_id=resample_id,
        data_transforms=data_transforms,
        build_train_file=build_train_file,
        build_test_file=build_test_file,
        attribute_file_path=attribute_file_path,
        att_max_shape=att_max_shape,
        benchmark_time=benchmark_time,
    )


def run_forecasting_experiment(
    train,
    y_test,
    estimator,
    results_path,
    estimator_name=None,
    dataset_name="",
    resample_id=None,
    data_transforms=None,
    build_test_file=True,
    build_train_file=False,
    attribute_file_path=None,
    att_max_shape=0,
    benchmark_time=True,
):
    """Run an experiment and save the results to file.

    Function to run a basic experiment for a
    <dataset>/<estimator>/<resample> combination and write the results to csv file(s)
    at a given location.

    Parameters
    ----------
    train : pd.DataFrame or np.array
        The series used to train the estimator.
    y_test : pd.DataFrame or np.array
        The series used to y_test the trained estimator.
    estimator : BaseForecaster
        Estimator to be used in the experiment.
    results_path : str
        Location of where to write results. Any required directories will be created.
    estimator_name : str or None, default=None
        Name of estimator used in writing results. If None, the name is taken from
        the estimator.
    dataset_name : str, default="N/A"
        Name of dataset.
    resample_id : int or None, default=None
        Indicates what random seed was used as a random_state for the estimator. Only
        used for the results file name.
    data_transforms : transformer, list of transformers or None, default=None
        Transformer(s) to apply to the data before running the experiment.
        If a list, the transformers are applied in order.
        If None, no transformation is applied.
        Calls fit_transform on the training data and transform on the test data.
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    """
    pass


def load_and_run_forecasting_experiment(
    problem_path,
    results_path,
    dataset,
    estimator,
    estimator_name=None,
    resample_id=None,
    write_attributes=False,
    att_max_shape=0,
    benchmark_time=True,
    overwrite=False,
):
    """Load a dataset and run an experiment.

    Function to load a dataset, run a basic experiment for a
    <dataset>/<estimator/<resample> combination, and write the results to csv file(s)
    at a given location.

    Parameters
    ----------
    problem_path : str
        Location of problem files, full path.
    results_path : str
        Location of where to write results. Any required directories will be created.
    dataset : str
        Name of problem. Files must be <problem_path>/<dataset>/<dataset>+"_TRAIN.csv",
        same for "_TEST.csv".
    estimator : BaseForecaster
        Estimator to be used in the experiment.
    estimator_name : str or None, default=None
        Name of estimator used in writing results. If None, the name is taken from
        the estimator.
    resample_id : int or None, default=None
        Indicates what random seed was used as a random_state for the estimator. Only
        used for the results file name.
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    overwrite : bool, default=False
        If set to False, this will only build results if there is not a result file
        already present. If True, it will overwrite anything already there.
    """
    tmpdir = tempfile.mkdtemp()
    dataset = load_forecasting(dataset, tmpdir)
    series = (
        dataset[dataset["series_name"] == series_name]["series_value"]
        .iloc[0]
        .to_numpy()
    )
    from aeon.transformations.series import TrainTestTransformer

    dataset = f"{dataset}_{series_name}"
    train, test = TrainTestTransformer().fit_transform(series)
    train = train.astype(float).to_numpy()
    test = test.astype(float).to_numpy()

    run_forecasting_experiment(
        train,
        test,
        estimator,
        results_path,
        estimator_name=estimator_name,
        dataset_name=dataset,
        resample_id=resample_id,
        attribute_file_path=attribute_file_path,
        att_max_shape=att_max_shape,
        benchmark_time=benchmark_time,
    )
