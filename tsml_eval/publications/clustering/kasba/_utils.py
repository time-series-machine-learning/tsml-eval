import csv
import os

import numpy as np
from aeon.datasets import load_from_ts_file
from aeon.transformations.collection import Normalizer

from tsml_eval.utils.resampling import stratified_resample_data


def load_results_from_csv(path: str) -> dict:
    """
    Load results from custom CSV format.

    Returns
    -------
    dict[str, dict[str, float]]
        { estimator: { dataset: score } }
    """
    with open(path) as f:
        reader = csv.reader(f)

        # Get header row
        header = next(reader)
        if header[0].lower().startswith("estimators"):
            estimators = [h.strip() for h in header[1:]]
        else:
            raise ValueError("First row must begin with 'Estimators:'")

        # Output structure
        results = {est: {} for est in estimators}

        # Process remaining rows
        for row in reader:
            if not row:
                continue

            dataset = row[0].strip()
            values = row[1:]

            if len(values) != len(estimators):
                raise ValueError(
                    f"Row for dataset '{dataset}' has {len(values)} values "
                    f"but expected {len(estimators)}"
                )

            for est, val in zip(estimators, values):
                # Convert scientific notation or normal float
                results[est][dataset] = float(val)

    return results


def results_to_matrix(results: dict):
    estimators = sorted(results.keys())

    datasets = sorted({d for est in estimators for d in results[est]})

    matrix = np.zeros((len(datasets), len(estimators)), dtype=float)

    for j, est in enumerate(estimators):
        for i, ds in enumerate(datasets):
            matrix[i, j] = results[est][ds]

    return matrix, datasets, estimators


def _parse_command_line_bool(s: str) -> bool:
    """Parse a boolean from common CLI strings."""
    t = s.strip().lower()
    if t in {"1", "true", "t", "yes", "y"}:
        return True
    if t in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(
        f"Invalid boolean value: {s!r}. Use one of: true/false/1/0/yes/no."
    )


def check_experiment_results_exist(
    model_name: str,
    dataset: str,
    combine_test_train: bool,
    path_to_results: str,
    resample_id: int = 0,
):
    """Check if the results of the experiment already exist.

    Parameters
    ----------
    model_name: str
        Name of the model.
    dataset: str
        Dataset name.
    combine_test_train: bool
        Boolean indicating if results for test train or combined should be checked.
    path_to_results: str
        Base path to the results.
    resample_id: int
        Integer indicating the resample id.


    Returns
    -------
    bool
        Boolean indicating if the results already exist.
    """
    path_to_train = os.path.join(
        path_to_results,
        model_name,
        "Predictions",
        dataset,
        f"trainResample{resample_id}.csv",
    )
    path_to_test = os.path.join(
        path_to_results,
        model_name,
        "Predictions",
        dataset,
        f"testResample{resample_id}.csv",
    )

    if combine_test_train:
        if os.path.exists(path_to_train):
            return True
    else:
        if os.path.exists(path_to_train) and os.path.exists(path_to_test):
            return True
    return False


def _normalize_data(X):
    scaler = Normalizer()
    return scaler.fit_transform(X)


def load_dataset_from_file(
    dataset_name: str,
    path_to_data: str,
    normalize: bool = True,
    combine_test_train: bool = False,
    resample_id: int | None = None,
):
    """Load dataset from file, optionally doing stratified resampling.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.
    path_to_data : str
        Path to the data.
    normalize : bool, default=True
        Whether to normalize the data.
    combine_test_train : bool, default=False
        Whether to combine the test and train data.
    resample_id : int or None, default=None
        If > 0 and combine_test_train is False, perform a stratified resample of the
        original TRAIN/TEST pair using this as the random seed.
        If 0 or None, use the original train/test split.
    """
    path_to_train_data = os.path.join(
        path_to_data, f"{dataset_name}/{dataset_name}_TRAIN.ts"
    )
    path_to_test_data = os.path.join(
        path_to_data, f"{dataset_name}/{dataset_name}_TEST.ts"
    )

    X_train, y_train = load_from_ts_file(path_to_train_data)
    X_test, y_test = load_from_ts_file(path_to_test_data)

    if not combine_test_train and resample_id is not None and resample_id > 0:
        X_train, y_train, X_test, y_test = stratified_resample_data(
            X_train,
            y_train,
            X_test,
            y_test,
            random_state=resample_id,
        )
    if combine_test_train:
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        if normalize:
            X = _normalize_data(X)
        return X, y, None, None
    else:
        if normalize:
            X_train = _normalize_data(X_train)
            X_test = _normalize_data(X_test)
        return X_train, y_train, X_test, y_test
