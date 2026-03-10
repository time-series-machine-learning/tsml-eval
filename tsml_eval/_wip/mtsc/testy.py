from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from aeon.datasets import load_classification
from aeon.classification.feature_based import FreshPRINCEClassifier
from aeon.classification.convolution_based import RocketClassifier
dataset = "BeijingPM10Quality_disc"

data_path = "C:/Temp/Classification/"
results_path = "C:/Temp/"
data_path = str(Path("~/Data").expanduser())
results_path = str(Path("~/Results").expanduser())
# dataset_path = path + data
# dataset_results = results + data




def write_test_resample_csv(
    results_path: str | Path,
    classifier_name: str,
    data_name: str,
    y_true,
    y_pred,
    y_proba,
    classes,
    resample_id: int = 0,
    split: str = "TEST",
    time_unit: str = "MILLISECONDS",
    description: str | None = None,
    classifier_params: dict | None = None,
    fit_time_ms: int = -1,
    predict_time_ms: int = -1,
    benchmark_time_ms: int = -1,
    memory_usage: int = -1,
    train_estimate_method: str = "N/A",
    train_estimate_time_ms: int = -1,
    fit_and_estimate_time_ms: int = -1,
) -> Path:
    """
    Write a tsml-style classification results file.

    Output file:
        <results_path>/<classifier_name>/Predictions/<data_name>/testResample{resample_id}.csv

    Parameters
    ----------
    results_path : str | Path
        Root results directory.
    classifier_name : str
        Name of classifier, e.g. "FreshPRINCE".
    data_name : str
        Dataset name, e.g. "Chinatown".
    y_true : array-like
        True class labels for the test set.
    y_pred : array-like
        Predicted class labels for the test set.
    y_proba : array-like of shape (n_cases, n_classes)
        Predicted probabilities. Column order must match `classes`.
    classes : array-like
        Class labels in the same order as the probability columns, usually `cls.classes_`.
    resample_id : int, default=0
        Resample identifier.
    split : str, default="TEST"
        Usually "TEST".
    time_unit : str, default="MILLISECONDS"
        Time unit label written into the file.
    description : str | None, default=None
        Optional free-text description.
    classifier_params : dict | None, default=None
        Parameters to write on line 2, usually `cls.get_params(deep=False)`.
    fit_time_ms, predict_time_ms, benchmark_time_ms, memory_usage,
    train_estimate_method, train_estimate_time_ms, fit_and_estimate_time_ms
        Summary metadata for line 3.

    Returns
    -------
    Path
        Full path to the written CSV file.
    """
    results_path = Path(results_path)
    out_dir = results_path / classifier_name / "Predictions" / data_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"testResample{resample_id}.csv"

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    classes = np.asarray(classes)

    if y_proba.ndim != 2:
        raise ValueError("y_proba must be a 2D array of shape (n_cases, n_classes).")

    if len(y_true) != len(y_pred) or len(y_true) != len(y_proba):
        raise ValueError("y_true, y_pred and y_proba must have the same number of cases.")

    if y_proba.shape[1] != len(classes):
        raise ValueError("Number of probability columns must match number of classes.")

    label_to_index = {label: i for i, label in enumerate(classes)}

    try:
        y_true_enc = [label_to_index[label] for label in y_true]
        y_pred_enc = [label_to_index[label] for label in y_pred]
    except KeyError as e:
        raise ValueError(f"Label {e.args[0]!r} found in y_true/y_pred but not in classes.") from e

    accuracy = float(np.mean(y_true == y_pred))
    n_classes = len(classes)

    if description is None:
        timestamp = time.strftime("%m/%d/%Y, %H:%M:%S")
        description = (
            f"Generated manually on {timestamp}. "
            f"Encoder dictionary: {label_to_index}"
        )

    if classifier_params is None:
        classifier_params = {}

    with out_file.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                data_name,
                classifier_name,
                split,
                resample_id,
                time_unit,
                description,
            ]
        )

        writer.writerow([classifier_params])

        writer.writerow(
            [
                accuracy,
                fit_time_ms,
                predict_time_ms,
                benchmark_time_ms,
                memory_usage,
                n_classes,
                train_estimate_method,
                train_estimate_time_ms,
                fit_and_estimate_time_ms,
            ]
        )

        for actual, pred, probs in zip(y_true_enc, y_pred_enc, y_proba):
            writer.writerow([actual, pred, ""] + list(map(float, probs)))

    return out_file



trainX, trainy = load_classification(name=dataset, extract_path=data_path,split="train")
testX, testy = load_classification(name=dataset, extract_path=data_path, split="test")
print("Train shape = ",trainX.shape)
print("Test shape = ",testX.shape)
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble


cls = TemporalDictionaryEnsemble(n_jobs=-1, time_limit_in_minutes=60)

start = time.perf_counter()
cls.fit(trainX, trainy)
fit_time_ms = (time.perf_counter() - start) * 1000

train_classes = np.unique(trainy)

start = time.perf_counter()
preds_proba = cls.predict_proba(testX)
predict_proba_time_ms = (time.perf_counter() - start) * 1000

preds = cls.classes_[np.argmax(preds_proba, axis=1)]

write_test_resample_csv(
    results_path=results_path,
    classifier_name="TDE",
    data_name=dataset,
    y_true=testy,
    y_pred=preds,
    y_proba=preds_proba,
    classes=train_classes,
    fit_time=int(fit_time_ms),
    predict_time=int(predict_proba_time_ms),
    resample_id=0,
)