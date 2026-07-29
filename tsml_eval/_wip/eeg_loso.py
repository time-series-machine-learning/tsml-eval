"""Run a leave-one-subject-out EEG classification experiment.

This is intentionally a small WIP entry point for the EEG case study. Subject
identifiers are read from the companion ``*_id_TRAIN.txt`` and
``*_id_TEST.txt`` files distributed with the archive data.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
from pathlib import Path

# Set these before importing NumPy, aeon or estimator implementations.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MPI_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

warnings.filterwarnings("ignore")

import numpy as np
from aeon.datasets import load_from_ts_file

from tsml_eval.experiments import (
    get_classifier_by_name,
    run_classification_experiment,
)


def _load_subject_ids(file_path: Path) -> np.ndarray:
    """Load one non-empty subject identifier per line."""
    with file_path.open(encoding="utf-8-sig") as file:
        subject_ids = [line.strip() for line in file if line.strip()]

    if not subject_ids:
        raise ValueError(f"No subject identifiers found in {file_path}")

    try:
        return np.asarray(subject_ids, dtype=np.int64)
    except ValueError as exc:
        raise ValueError(
            f"Subject identifiers in {file_path} must be integers."
        ) from exc


def _subject_files(data_path: Path, dataset: str) -> tuple[Path, Path, Path, Path]:
    dataset_path = data_path / dataset
    return (
        dataset_path / f"{dataset}_TRAIN.ts",
        dataset_path / f"{dataset}_TEST.ts",
        dataset_path / f"{dataset}_id_TRAIN.txt",
        dataset_path / f"{dataset}_id_TEST.txt",
    )


def _validate_subject_files(
    data_path: Path, dataset: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_file, test_file, train_id_file, test_id_file = _subject_files(
        data_path, dataset
    )
    for file_path in (train_file, test_file, train_id_file, test_id_file):
        if not file_path.is_file() or file_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty input file: {file_path}")

    train_ids = _load_subject_ids(train_id_file)
    test_ids = _load_subject_ids(test_id_file)
    subjects = np.unique(np.concatenate((train_ids, test_ids)))

    overlap = np.intersect1d(np.unique(train_ids), np.unique(test_ids))
    if overlap.size:
        raise ValueError(
            "The archive TRAIN and TEST collections unexpectedly share subject "
            f"identifiers: {overlap.tolist()}"
        )

    return train_ids, test_ids, subjects


def _copy_cases(
    X_parts: tuple[np.ndarray, np.ndarray],
    y_parts: tuple[np.ndarray, np.ndarray],
    id_parts: tuple[np.ndarray, np.ndarray],
    held_subject: int,
    *,
    held_out: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy either the held subject or its complement into contiguous arrays."""
    indices = tuple(
        np.flatnonzero(ids == held_subject if held_out else ids != held_subject)
        for ids in id_parts
    )
    n_cases = sum(index.size for index in indices)
    if n_cases == 0:
        split = "test" if held_out else "train"
        raise ValueError(f"The LOSO {split} split contains no cases.")

    case_shape = X_parts[0].shape[1:]
    if any(X.ndim != 3 or X.shape[1:] != case_shape for X in X_parts):
        raise ValueError(
            "OpenCloseFist LOSO expects equal-length 3D collections with "
            "matching channel and time dimensions."
        )

    X_out = np.empty(
        (n_cases, *case_shape),
        dtype=np.result_type(*(X.dtype for X in X_parts)),
    )
    y_out = np.empty(n_cases, dtype=np.result_type(*(y.dtype for y in y_parts)))

    offset = 0
    for X, y, selected in zip(X_parts, y_parts, indices):
        next_offset = offset + selected.size
        np.take(X, selected, axis=0, out=X_out[offset:next_offset])
        np.take(y, selected, axis=0, out=y_out[offset:next_offset])
        offset = next_offset

    return X_out, y_out


def load_loso_split(
    data_path: Path,
    dataset: str,
    held_subject: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the archive splits and construct one LOSO split."""
    train_file, test_file, _, _ = _subject_files(data_path, dataset)
    train_ids, test_ids, subjects = _validate_subject_files(data_path, dataset)

    if held_subject not in subjects:
        raise ValueError(
            f"Subject {held_subject} is not present. Available subjects are "
            f"{subjects[0]} to {subjects[-1]} ({subjects.size} total)."
        )

    X_archive_train, y_archive_train = load_from_ts_file(train_file)
    X_archive_test, y_archive_test = load_from_ts_file(test_file)

    if len(y_archive_train) != train_ids.size:
        raise ValueError(
            f"{train_file.name} contains {len(y_archive_train)} cases but "
            f"{train_ids.size} TRAIN subject identifiers were supplied."
        )
    if len(y_archive_test) != test_ids.size:
        raise ValueError(
            f"{test_file.name} contains {len(y_archive_test)} cases but "
            f"{test_ids.size} TEST subject identifiers were supplied."
        )

    X_parts = (X_archive_train, X_archive_test)
    y_parts = (np.asarray(y_archive_train), np.asarray(y_archive_test))
    id_parts = (train_ids, test_ids)

    X_train, y_train = _copy_cases(
        X_parts, y_parts, id_parts, held_subject, held_out=False
    )
    X_test, y_test = _copy_cases(
        X_parts, y_parts, id_parts, held_subject, held_out=True
    )

    del X_archive_train, X_archive_test, y_archive_train, y_archive_test
    del X_parts, y_parts
    gc.collect()

    if np.unique(y_train).size < 2:
        raise ValueError("The LOSO training split contains fewer than two classes.")

    return X_train, y_train, X_test, y_test, subjects


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one leave-one-subject-out EEG experiment."
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("results_path", type=Path)
    parser.add_argument("classifier")
    parser.add_argument("held_subject", type=int)
    parser.add_argument("--dataset", default="OpenCloseFist")
    parser.add_argument(
        "--classifier-name",
        help="Result-directory name; defaults to the classifier factory name.",
    )
    parser.add_argument(
        "--no-train-file",
        action="store_true",
        help="Do not generate train predictions (not suitable for HC2-from-file).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the files and subject IDs without loading the .ts data.",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    config = _parse_args(args)
    classifier_name = config.classifier_name or config.classifier
    result_dataset = f"{config.dataset}LOSO"

    train_ids, test_ids, subjects = _validate_subject_files(
        config.data_path, config.dataset
    )
    if config.held_subject not in subjects:
        raise ValueError(
            f"Subject {config.held_subject} is not one of "
            f"{subjects[0]}..{subjects[-1]}."
        )

    print(f"Dataset:             {config.dataset}")
    print(f"Result dataset:      {result_dataset}")
    print(f"Classifier factory:  {config.classifier}")
    print(f"Classifier result:   {classifier_name}")
    print(f"Held subject:        {config.held_subject}")
    print(f"Archive TRAIN cases: {train_ids.size}")
    print(f"Archive TEST cases:  {test_ids.size}")
    print(f"Subjects:            {subjects.size}")

    if config.validate_only:
        print("Subject-file validation succeeded.")
        return

    X_train, y_train, X_test, y_test, _ = load_loso_split(
        config.data_path,
        config.dataset,
        config.held_subject,
    )
    print(f"LOSO train shape:    {X_train.shape}")
    print(f"LOSO test shape:     {X_test.shape}")
    print(
        "LOSO train classes:  "
        f"{dict(zip(*np.unique(y_train, return_counts=True)))}"
    )
    print(
        "LOSO test classes:   "
        f"{dict(zip(*np.unique(y_test, return_counts=True)))}"
    )

    classifier = get_classifier_by_name(
        config.classifier,
        random_state=config.held_subject,
        n_jobs=1,
    )
    run_classification_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier,
        config.results_path,
        classifier_name=classifier_name,
        dataset_name=result_dataset,
        resample_id=config.held_subject,
        build_train_file=not config.no_train_file,
        benchmark_time=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
