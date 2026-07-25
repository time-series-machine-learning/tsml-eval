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

import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime
from importlib import import_module, metadata
from pathlib import Path

import numpy as np
import pandas as pd
from aeon.benchmarking.metrics.clustering import clustering_accuracy_score
from aeon.classification import BaseClassifier
from aeon.clustering import BaseClusterer
from aeon.forecasting import BaseForecaster
from aeon.regression.base import BaseRegressor
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.utils.validation.collection import get_n_cases
from sklearn import preprocessing
from sklearn.base import BaseEstimator, is_classifier, is_clusterer, is_regressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import cross_val_predict

from tsml_eval.estimators.classification._sklearn import SklearnToAeonClassifier
from tsml_eval.estimators.clustering._sklearn import SklearnToAeonClusterer
from tsml_eval.estimators.regression._sklearn import SklearnToAeonRegressor
from tsml_eval.utils.datasets import load_experiment_data
from tsml_eval.utils.experiments import (
    _check_existing_results,
    estimator_attributes_to_file,
    timing_benchmark,
)
from tsml_eval.utils.memory_recorder import record_max_memory
from tsml_eval.utils.resampling import resample_data, stratified_resample_data
from tsml_eval.utils.results_writing import (
    write_classification_results,
    write_clustering_results,
    write_forecasting_results,
    write_regression_results,
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
    # aeon classifier
    if isinstance(classifier, BaseClassifier):
        if not ignore_custom_train_estimate and classifier.get_tag(
            "capability:train_estimate", False, False
        ):
            use_fit_predict = True
    elif isinstance(classifier, BaseEstimator) and is_classifier(classifier):
        is_sklearn = True
        if _check_soft_dependencies("tsml", severity="none"):
            from tsml.base import BaseTimeSeriesEstimator

            # tsml classifier
            if isinstance(classifier, BaseTimeSeriesEstimator):
                is_sklearn = False

        # assumed sklearn classifier
        if is_sklearn:
            classifier = SklearnToAeonClassifier(
                classifier=classifier,
                pad_unequal=True,
                concatenate_channels=True,
                random_state=(
                    classifier.random_state
                    if hasattr(classifier, "random_state")
                    else None
                ),
            )
    else:
        raise TypeError("classifier must be an aeon, tsml or sklearn classifier.")

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
    run_metadata = _get_classification_run_metadata(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classifier_name=classifier_name,
        dataset_name=dataset_name,
        resample_id=resample_id,
        benchmark_time=benchmark_time,
    )

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

    second = _get_estimator_parameter_info(classifier, run_metadata)

    if build_train_file:
        cv_size = 10
        start = int(round(time.time() * 1000))
        if use_fit_predict:
            train_probs = classifier.fit_predict_proba(X_train, y_train)
            needs_fit = False
            if hasattr(classifier, "fit_time_millis_"):
                fit_time = classifier.fit_time_millis_
            fit_and_train_time = int(round(time.time() * 1000)) - start
        else:
            _, counts = np.unique(y_train, return_counts=True)
            cv_size = min(cv_size, max(2, int(np.min(counts))))

            train_probs = cross_val_predict(
                classifier, X_train, y=y_train, cv=cv_size, method="predict_proba"
            )
            train_time = int(round(time.time() * 1000)) - start

        train_preds = np.unique(y_train)[np.argmax(train_probs, axis=1)]
        train_acc = accuracy_score(y_train, train_preds)
        second = _get_estimator_parameter_info(classifier, run_metadata)

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
            if hasattr(classifier, "fit_time_millis_"):
                fit_time = classifier.fit_time_millis_

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
        second = _get_estimator_parameter_info(classifier, run_metadata)

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


def _get_estimator_parameter_info(estimator, run_metadata=None):
    """Return parameters plus optional fitted experiment metadata on one line."""
    parameter_info = str(estimator.get_params())
    metadata_fields = {}
    if run_metadata:
        metadata_fields["run"] = run_metadata

    metadata_getter = getattr(estimator, "get_experiment_metadata", None)
    if callable(metadata_getter):
        estimator_metadata = metadata_getter()
        if estimator_metadata:
            metadata_fields.update(estimator_metadata)

    if metadata_fields:
        parameter_info += " | experiment_metadata=" + json.dumps(
            metadata_fields,
            separators=(",", ":"),
            sort_keys=True,
        )
    return parameter_info.replace("\n", " ").replace("\r", " ")


def _get_classification_run_metadata(
    X_train,
    y_train,
    X_test,
    y_test,
    classifier_name,
    dataset_name,
    resample_id,
    benchmark_time,
):
    """Collect compact data, software and execution provenance for a result."""
    return {
        "experiment": {
            "classifier": classifier_name,
            "dataset": dataset_name,
            "resample_id": resample_id,
            "benchmark_enabled": bool(benchmark_time),
            "command": list(sys.argv),
        },
        "data": {
            "train": _collection_metadata(X_train),
            "test": _collection_metadata(X_test),
            "train_class_counts": _class_counts(y_train),
            "test_class_counts": _class_counts(y_test),
        },
        "environment": _environment_metadata(),
    }


def _collection_metadata(X):
    """Summarise an aeon collection without serialising the collection itself."""
    if isinstance(X, np.ndarray):
        return {
            "container": "ndarray",
            "shape": [int(value) for value in X.shape],
            "dtype": str(X.dtype),
        }

    n_cases = int(get_n_cases(X))
    channel_counts = []
    timepoint_counts = []
    for case in X:
        case_array = np.asarray(case)
        if case_array.ndim == 1:
            channel_counts.append(1)
            timepoint_counts.append(int(case_array.shape[0]))
        else:
            channel_counts.append(int(case_array.shape[-2]))
            timepoint_counts.append(int(case_array.shape[-1]))

    return {
        "container": type(X).__name__,
        "n_cases": n_cases,
        "n_channels_min": min(channel_counts, default=0),
        "n_channels_max": max(channel_counts, default=0),
        "n_timepoints_min": min(timepoint_counts, default=0),
        "n_timepoints_max": max(timepoint_counts, default=0),
        "equal_length": len(set(timepoint_counts)) <= 1,
    }


def _class_counts(y):
    """Return JSON-safe class frequencies."""
    labels, counts = np.unique(y, return_counts=True)
    return {
        str(label.item() if hasattr(label, "item") else label): int(count)
        for label, count in zip(labels, counts)
    }


def _environment_metadata():
    """Return software, Git, scheduler and thread provenance."""
    environment = {
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "logical_cpus": os.cpu_count(),
        "packages": {},
    }

    package_specs = (
        ("tsml_eval", "tsml-eval"),
        ("aeon", "aeon"),
        ("aeon_neuro", "aeon-neuro"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("numba", "numba"),
    )
    repositories = {}
    for module_name, distribution_name in package_specs:
        package_info = _package_metadata(module_name, distribution_name)
        if package_info is None:
            continue
        environment["packages"][distribution_name] = package_info

        git_info = _git_metadata(package_info.get("path"))
        if git_info is not None:
            repositories[distribution_name] = git_info

    if repositories:
        environment["git"] = repositories

    variable_names = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_NODELIST",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_CPU",
        "SLURM_MEM_PER_NODE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
        "CUDA_VISIBLE_DEVICES",
    )
    variables = {
        name: os.environ[name] for name in variable_names if name in os.environ
    }
    if variables:
        environment["variables"] = variables
    return environment


def _package_metadata(module_name, distribution_name):
    """Return the installed version and imported source path, if available."""
    try:
        module = import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None

    try:
        version = metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "unknown")

    module_path = getattr(module, "__file__", None)
    return {
        "version": str(version),
        "path": str(Path(module_path).resolve()) if module_path else None,
    }


def _git_metadata(module_path):
    """Return commit, branch and tracked dirty state for a source checkout."""
    if module_path is None:
        return None

    path = Path(module_path).resolve().parent
    repository = next(
        (parent for parent in (path, *path.parents) if (parent / ".git").exists()),
        None,
    )
    if repository is None:
        return None

    def _git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
        branch = _git("branch", "--show-current") or "detached"
        status = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return None

    return {
        "root": str(repository),
        "commit": commit,
        "branch": branch,
        "tracked_changes": bool(status),
    }


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


def run_regression_experiment(
    X_train: np.ndarray | list,
    y_train: np.ndarray,
    X_test: np.ndarray | list,
    y_test: np.ndarray,
    regressor,
    results_path,
    regressor_name=None,
    dataset_name="",
    resample_id=None,
    data_transforms=None,
    build_test_file=True,
    build_train_file=False,
    ignore_custom_train_estimate=False,
    attribute_file_path=None,
    att_max_shape=0,
    benchmark_time=True,
):
    """Run a regression experiment and save the results to file.

    Function to run a basic regression experiment for a
    <dataset>/<regressor>/<resample> combination and write the results to csv file(s)
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
    regressor : BaseRegressor
        Regressor to be used in the experiment.
    results_path : str
        Location of where to write results. Any required directories will be created.
    regressor_name : str or None, default=None
        Name of regressor used in writing results. If None, the name is taken from
        the regressor.
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
        Whether to generate test files or not. If the regressor can generate its own
        train predictions, the classifier will be built but no file will be output.
    build_train_file : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the regressor can produce its
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

    if regressor_name is None:
        regressor_name = type(regressor).__name__

    use_fit_predict = False
    # aeon regressor
    if isinstance(regressor, BaseRegressor):
        if not ignore_custom_train_estimate and regressor.get_tag(
            "capability:train_estimate", False, False
        ):
            use_fit_predict = True
    elif isinstance(regressor, BaseEstimator) and is_regressor(regressor):
        is_sklearn = True
        if _check_soft_dependencies("tsml", severity="none"):
            from tsml.base import BaseTimeSeriesEstimator

            # tsml regressor
            if isinstance(regressor, BaseTimeSeriesEstimator):
                is_sklearn = False

        # assumed sklearn regressor
        if is_sklearn:
            regressor = SklearnToAeonRegressor(
                regressor=regressor,
                pad_unequal=True,
                concatenate_channels=True,
                random_state=(
                    regressor.random_state
                    if hasattr(regressor, "random_state")
                    else None
                ),
            )
    else:
        raise TypeError("regressor must be a tsml, aeon or sklearn regressor.")

    if data_transforms is not None:
        if not isinstance(data_transforms, list):
            data_transforms = [data_transforms]

        for transform in data_transforms:
            X_train = transform.fit_transform(X_train, y_train)
            X_test = transform.transform(X_test, y_test)

    needs_fit = True
    fit_time = -1
    mem_usage = -1
    benchmark = -1
    train_time = -1
    fit_and_train_time = -1

    if benchmark_time:
        benchmark = timing_benchmark(random_state=resample_id)

    first_comment = (
        "Generated by run_regression_experiment on "
        f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}"
    )

    second = str(regressor.get_params()).replace("\n", " ").replace("\r", " ")

    if build_train_file:
        cv_size = min(10, len(y_train))
        start = int(round(time.time() * 1000))
        if use_fit_predict:
            train_preds = regressor.fit_predict(X_train, y_train)
            needs_fit = False
            fit_and_train_time = int(round(time.time() * 1000)) - start
        else:
            train_preds = cross_val_predict(regressor, X_train, y=y_train, cv=cv_size)
            train_time = int(round(time.time() * 1000)) - start

        train_mse = mean_squared_error(y_train, train_preds)

        write_regression_results(
            train_preds,
            y_train,
            regressor_name,
            dataset_name,
            results_path,
            full_path=False,
            first_line_regressor_name=f"{regressor_name} ({type(regressor).__name__})",
            split="TRAIN",
            resample_id=resample_id,
            time_unit="MILLISECONDS",
            first_line_comment=first_comment,
            parameter_info=second,
            mse=train_mse,
            fit_time=fit_time,
            predict_time=-1,
            benchmark_time=benchmark,
            memory_usage=mem_usage,
            train_estimate_method="Custom" if use_fit_predict else f"{cv_size}F-CV",
            train_estimate_time=train_time,
            fit_and_estimate_time=fit_and_train_time,
        )

    if build_test_file:
        if needs_fit:
            mem_usage, fit_time = record_max_memory(
                regressor.fit,
                args=(X_train, y_train),
                interval=MEMRECORD_INTERVAL,
                return_func_time=True,
            )
            fit_time += int(round(getattr(regressor, "fit_time_milli_", 0)))

        if attribute_file_path is not None:
            estimator_attributes_to_file(
                regressor, attribute_file_path, max_list_shape=att_max_shape
            )

        start = int(round(time.time() * 1000))
        test_preds = regressor.predict(X_test)
        test_time = (int(round(time.time() * 1000)) - start) + int(
            round(getattr(regressor, "predict_time_milli_", 0))
        )

        test_mse = mean_squared_error(y_test, test_preds)

        write_regression_results(
            test_preds,
            y_test,
            regressor_name,
            dataset_name,
            results_path,
            full_path=False,
            first_line_regressor_name=f"{regressor_name} ({type(regressor).__name__})",
            split="TEST",
            resample_id=resample_id,
            time_unit="MILLISECONDS",
            first_line_comment=first_comment,
            parameter_info=second,
            mse=test_mse,
            fit_time=fit_time,
            predict_time=test_time,
            benchmark_time=benchmark,
            memory_usage=mem_usage,
            train_estimate_method="N/A",
            train_estimate_time=-1,
            fit_and_estimate_time=fit_and_train_time,
        )


def load_and_run_regression_experiment(
    problem_path,
    results_path,
    dataset,
    regressor,
    regressor_name=None,
    resample_id=0,
    data_transforms=None,
    build_train_file=False,
    write_attributes=False,
    att_max_shape=0,
    benchmark_time=True,
    overwrite=False,
    predefined_resample=False,
):
    """Load a dataset and run a regression experiment.

    Function to load a dataset, run a basic regression experiment for a
    <dataset>/<regressor>/<resample> combination, and write the results to csv file(s)
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
    regressor : BaseRegressor
        Regressor to be used in the experiment.
    regressor_name : str or None, default=None
        Name of regressor used in writing results. If None, the name is taken from
        the regressor.
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
        cross-validation on the train data and saves. If the regressor can produce its
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
    if regressor_name is None:
        regressor_name = type(regressor).__name__

    build_test_file, build_train_file = _check_existing_results(
        results_path,
        regressor_name,
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
        X_train, y_train, X_test, y_test = resample_data(
            X_train, y_train, X_test, y_test, random_state=resample_id
        )

    if write_attributes:
        attribute_file_path = f"{results_path}/{regressor_name}/Workspace/{dataset}/"
    else:
        attribute_file_path = None

    # Ensure labels are floats
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    run_regression_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        regressor,
        results_path,
        regressor_name=regressor_name,
        dataset_name=dataset,
        resample_id=resample_id,
        data_transforms=data_transforms,
        build_test_file=build_test_file,
        build_train_file=build_train_file,
        attribute_file_path=attribute_file_path,
        att_max_shape=att_max_shape,
        benchmark_time=benchmark_time,
    )


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

    # aeon clusterer
    if isinstance(clusterer, BaseClusterer):
        pass
    elif isinstance(clusterer, BaseEstimator) and is_clusterer(clusterer):
        is_sklearn = True
        if _check_soft_dependencies("tsml", severity="none"):
            from tsml.base import BaseTimeSeriesEstimator

            # tsml clusterer
            if isinstance(clusterer, BaseTimeSeriesEstimator):
                is_sklearn = False

        # assumed sklearn clusterer
        if is_sklearn:
            clusterer = SklearnToAeonClusterer(
                clusterer=clusterer,
                pad_unequal=True,
                concatenate_channels=True,
                random_state=(
                    clusterer.random_state
                    if hasattr(clusterer, "random_state")
                    else None
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
    fit_time += int(round(getattr(clusterer, "fit_time_milli_", 0)))

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
    test,
    forecaster,
    results_path,
    forecaster_name=None,
    dataset_name="N/A",
    random_seed=None,
    attribute_file_path=None,
    att_max_shape=0,
    benchmark_time=True,
):
    """Run a forecasting experiment and save the results to file.

    Function to run a basic forecasting experiment for a
    <dataset>/<forecaster>/<resample> combination and write the results to csv file(s)
    at a given location.

    Parameters
    ----------
    train : pd.DataFrame or np.array
        The series used to train the forecaster.
    test : pd.DataFrame or np.array
        The series used to test the trained forecaster.
    forecaster : BaseForecaster
        Regressor to be used in the experiment.
    results_path : str
        Location of where to write results. Any required directories will be created.
    forecaster_name : str or None, default=None
        Name of forecaster used in writing results. If None, the name is taken from
        the forecaster.
    dataset_name : str, default="N/A"
        Name of dataset.
    random_seed : int or None, default=None
        Indicates what random seed was used as a random_state for the forecaster. Only
        used for the results file name.
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    """
    if not isinstance(forecaster, BaseForecaster):
        raise TypeError("forecaster must be an aeon forecaster.")

    if forecaster_name is None:
        forecaster_name = type(forecaster).__name__

    benchmark = -1
    if benchmark_time:
        benchmark = timing_benchmark(random_state=random_seed)

    first_comment = (
        "Generated by run_forecasting_experiment on "
        f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}"
    )

    second = str(forecaster.get_params()).replace("\n", " ").replace("\r", " ")

    mem_usage, fit_time = record_max_memory(
        forecaster.fit,
        args=(train,),
        interval=MEMRECORD_INTERVAL,
        return_func_time=True,
    )
    fit_time += int(round(getattr(forecaster, "fit_time_milli_", 0)))

    if attribute_file_path is not None:
        estimator_attributes_to_file(
            forecaster, attribute_file_path, max_list_shape=att_max_shape
        )

    start = int(round(time.time() * 1000))
    test_preds = forecaster.predict(np.arange(1, len(test) + 1))
    test_time = (
        int(round(time.time() * 1000))
        - start
        + int(round(getattr(forecaster, "predict_time_milli_", 0)))
    )
    test_preds = test_preds.flatten()

    test_mape = mean_absolute_percentage_error(test, test_preds)

    write_forecasting_results(
        test_preds,
        test,
        forecaster_name,
        dataset_name,
        results_path,
        full_path=False,
        first_line_forecaster_name=f"{forecaster_name} ({type(forecaster).__name__})",
        split="TEST",
        random_seed=random_seed,
        time_unit="MILLISECONDS",
        first_line_comment=first_comment,
        parameter_info=second,
        mape=test_mape,
        fit_time=fit_time,
        predict_time=test_time,
        benchmark_time=benchmark,
        memory_usage=mem_usage,
    )


def load_and_run_forecasting_experiment(
    problem_path,
    results_path,
    dataset,
    forecaster,
    forecaster_name=None,
    random_seed=None,
    write_attributes=False,
    att_max_shape=0,
    benchmark_time=True,
    overwrite=False,
):
    """Load a dataset and run a regression experiment.

    Function to load a dataset, run a basic regression experiment for a
    <dataset>/<regressor/<resample> combination, and write the results to csv file(s)
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
    forecaster : BaseForecaster
        Regressor to be used in the experiment.
    forecaster_name : str or None, default=None
        Name of forecaster used in writing results. If None, the name is taken from
        the forecaster.
    random_seed : int or None, default=None
        Indicates what random seed was used as a random_state for the forecaster. Only
        used for the results file name.
    benchmark_time : bool, default=True
        Whether to benchmark the hardware used with a simple function and write the
        results. This will typically take ~2 seconds, but is hardware dependent.
    overwrite : bool, default=False
        If set to False, this will only build results if there is not a result file
        already present. If True, it will overwrite anything already there.
    """
    if forecaster_name is None:
        forecaster_name = type(forecaster).__name__

    build_test_file, _ = _check_existing_results(
        results_path,
        forecaster_name,
        dataset,
        random_seed,
        overwrite,
        True,
        False,
    )

    if not build_test_file:
        warnings.warn("All files exist and not overwriting, skipping.", stacklevel=1)
        return

    if write_attributes:
        attribute_file_path = f"{results_path}/{forecaster_name}/Workspace/{dataset}/"
    else:
        attribute_file_path = None

    train = pd.read_csv(
        f"{problem_path}/{dataset}/{dataset}_TRAIN.csv", index_col=0
    ).squeeze("columns")
    train = train.astype(float).to_numpy()
    test = pd.read_csv(
        f"{problem_path}/{dataset}/{dataset}_TEST.csv", index_col=0
    ).squeeze("columns")
    test = test.astype(float).to_numpy()

    run_forecasting_experiment(
        train,
        test,
        forecaster,
        results_path,
        forecaster_name=forecaster_name,
        dataset_name=dataset,
        random_seed=random_seed,
        attribute_file_path=attribute_file_path,
        att_max_shape=att_max_shape,
        benchmark_time=benchmark_time,
    )
