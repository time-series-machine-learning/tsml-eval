import csv
import os

import numpy as np
from aeon.datasets import load_from_ts_file
from aeon.transformations.collection import Normalizer

from tsml_eval.utils.resampling import stratified_resample_data


def load_results_from_csv(path: str) -> dict[str, dict[str, float]]:
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
        results: dict[str, dict[str, float]] = {est: {} for est in estimators}

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
                results[est][dataset] = float(val)

    return results


def results_to_matrix(
    results: dict[str, dict[str, float]],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Convert results dict into a (n_datasets, n_estimators) matrix.

    Returns
    -------
    matrix : np.ndarray
        Shape (n_datasets, n_estimators)
    datasets : list[str]
    estimators : list[str]
    """
    estimators: list[str] = sorted(results.keys())
    datasets: list[str] = sorted({d for est in estimators for d in results[est]})

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
) -> bool:
    """
    Check if the results of the experiment already exist.

    Returns
    -------
    bool
        True if results already exist.
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
        return os.path.exists(path_to_train)
    else:
        return os.path.exists(path_to_train) and os.path.exists(path_to_test)


def _normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalize time series collection data."""
    scaler = Normalizer()
    return scaler.fit_transform(X)


def load_dataset_from_file(
    dataset_name: str,
    path_to_data: str,
    normalize: bool = True,
    combine_test_train: bool = False,
    resample_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Load dataset from file, optionally doing stratified resampling.

    Returns
    -------
    (X_train, y_train, X_test, y_test)
        Or (X, y, None, None) if combine_test_train=True
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
