"""Functions to perform machine learning/data mining experiments.

Results are saved a standardised format used by tsml.
"""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = [
    "load_and_run_classification_experiment_ml",
]

import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import cross_val_predict

from tsml_eval.estimators import (
    SklearnToTsmlClassifier,
    SklearnToTsmlClusterer,
    SklearnToTsmlRegressor,
)
from tsml_eval.time_series_meta_learning.base_torch import BaseDeepClassifier_Pytorch
from tsml_eval.time_series_meta_learning.imbalanced_classification_ml.data import (
    Task_Data,
)
from tsml_eval.utils.datasets import load_experiment_data
from tsml_eval.utils.experiments import (
    _check_existing_results,
    estimator_attributes_to_file,
    timing_benchmark,
)
from tsml_eval.utils.memory_recorder import record_max_memory
from tsml_eval.utils.oversampling_methods import SMOTE_FAMILY
from tsml_eval.utils.resampling import (
    make_imbalance,
    resample_data,
    stratified_resample_data,
)
from tsml_eval.utils.results_writing import (
    write_classification_results,
    write_clustering_results,
    write_forecasting_results,
    write_regression_results,
)

if os.getenv("MEMRECORD_INTERVAL") is not None:  # pragma: no cover
    TEMP = os.getenv("MEMRECORD_INTERVAL")
    MEMRECORD_INTERVAL = float(TEMP) if isinstance(TEMP, str) else 5.0
else:
    MEMRECORD_INTERVAL = 5.0


def load_and_run_classification_experiment_ml(
    problem_path,
    results_path,
    dataset,
    datasetlists,
    classifier,
    row_normalise=False,
    classifier_name=None,
    resample_id=0,
    build_train_file=False,
    write_attributes=False,
    att_max_shape=0,
    benchmark_time=True,
    overwrite=False,
    predefined_resample=False,
    test_oversampling_methods=None,
    imbalance_ratio=None,
    ignore_custom_train_estimate=False,
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
    classifier : BaseDeepClassifier_Pytorch
        Classifier to be used in the experiment.
    row_normalise : bool, default=False
        Whether to normalise the data rows (time series) prior to fitting and
        predicting.
    classifier_name : str or None, default=None
        Name of classifier used in writing results. If None, the name is taken from
        the classifier.
    resample_id : int, default=0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
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
    test_smote_family_resample : bool, default=False
    imbalance_ratio : int, default=None
        used to create imbalance data, the value is the ratio of the majority class to
        the minority class
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

    # Meta_training starts here

    if write_attributes:
        attribute_file_path = f"{results_path}/{classifier_name}/Workspace/{dataset}/"
    else:
        attribute_file_path = None

    dataset_name = dataset
    data_utils = Task_Data(
        problem_path=problem_path,
        dataset=dataset,
        resample_id=resample_id,
        predefined_resample=predefined_resample,
        datasetlists=datasetlists,
        K_support=5,
        K_Query=5,
    )
    X_train, y_train, X_test, y_test = data_utils.get_meta_test_task(
        imbalance_ratio=imbalance_ratio
    )
    if classifier_name is None:
        classifier_name = type(classifier).__name__

    use_fit_predict = False
    if isinstance(classifier, BaseDeepClassifier_Pytorch):
        if not ignore_custom_train_estimate and classifier.get_tag(
            "capability:train_estimate", False, False
        ):
            use_fit_predict = True
    else:
        raise TypeError("classifier must be a tsml, aeon or sklearn classifier.")

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
    )

    second = str(classifier.get_params()).replace("\n", " ").replace("\r", " ")

    if build_train_file:
        cv_size = 10
        start = int(round(time.time() * 1000))
        if classifier.meta_train_model:
            classifier.meta_train()
        if use_fit_predict:
            train_probs = classifier.fit_predict_proba(X_train, y_train)
            needs_fit = False
            fit_and_train_time = int(round(time.time() * 1000)) - start
        else:
            _, counts = np.unique(y_train, return_counts=True)
            n_classes = len(counts)
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
