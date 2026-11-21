"""Within-subject cross-validation with contiguous folds.

For one subject of a dataset, the subject's cases (train and test combined) are put in
recording order and cut into ``n_folds`` contiguous blocks. Each block is the test
set once, with the other blocks as training data, so trials close in time stay in the
same fold. One file per subject records every case's fold, label, prediction and
class probabilities, so per-subject scores (and their mean and spread over subjects)
can be computed without a results file per fold.
"""

__all__ = ["run_within_subject_experiment", "load_case_order", "subject_folds"]

import csv
import os
import time
import warnings
from datetime import datetime

import numpy as np
from sklearn.base import clone

from tsml_eval.utils.datasets import load_experiment_data, load_subject_ids


def _natural(value):
    """Sort key that orders numbers numerically and puts them before text."""
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


def load_case_order(problem_path, dataset):
    """Return a recording-order key for each case, train cases first.

    Read from ``cases_train.tsv`` and ``cases_test.tsv`` in the dataset folder, whose
    ``row`` column gives each case's position in its .ts file. The key is (session,
    run, start), with start the ``start_sample`` column, else ``start_time_seconds``,
    else the row.

    Parameters
    ----------
    problem_path : str
        Path to the problem folder.
    dataset : str
        Name of the dataset.

    Returns
    -------
    keys : list of tuple
        Sortable recording-order key of each case.
    """
    keys = []
    for split in ("train", "test"):
        path = f"{problem_path}/{dataset}/cases_{split}.tsv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is needed for the trial order")
        with open(path, encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
        rows.sort(key=lambda r: int(r["row"]))
        if [int(r["row"]) for r in rows] != list(range(len(rows))):
            raise ValueError(f"{path}: rows are not 0 to {len(rows) - 1}")
        for r in rows:
            start = r.get("start_sample") or r.get("start_time_seconds") or r["row"]
            keys.append(
                (
                    _natural(r.get("session_id", "")),
                    _natural(r.get("run_id", "")),
                    _natural(start),
                )
            )
    return keys


def subject_folds(groups, order, subject_index, n_folds=10):
    """Return the subject's ID and its cases as ``n_folds`` contiguous folds.

    Parameters
    ----------
    groups : np.ndarray
        Subject ID of each case, train cases first.
    order : list of tuple
        Recording-order key of each case, as from ``load_case_order``.
    subject_index : int
        Position of the subject among the sorted unique IDs.
    n_folds : int, default=10
        Number of folds.

    Returns
    -------
    subject : str
        The subject's ID.
    folds : list of np.ndarray
        Case indices of each fold, each in recording order.
    """
    subjects = np.unique(groups)
    if not 0 <= subject_index < len(subjects):
        raise ValueError(
            f"subject_index {subject_index} out of range for {len(subjects)} subjects"
        )
    subject = subjects[subject_index]
    cases = sorted(np.flatnonzero(groups == subject), key=lambda i: order[i])
    if len(cases) < n_folds:
        raise ValueError(
            f"subject {subject} has {len(cases)} cases, fewer than {n_folds} folds"
        )
    return str(subject), [np.asarray(f) for f in np.array_split(cases, n_folds)]


def _subset(X, idx):
    return X[idx] if isinstance(X, np.ndarray) else [X[i] for i in idx]


def run_within_subject_experiment(
    problem_path,
    results_path,
    dataset,
    classifier,
    classifier_name=None,
    subject_index=0,
    n_folds=10,
    overwrite=False,
):
    """Run contiguous-fold cross-validation within one subject and write its file.

    The file is ``<results_path>/<classifier_name>/WithinSubject/<dataset>/
    subject<subject_index>.csv``: a ``#`` comment line of settings, then one row per
    case with its fold, source split and row, recording position, true label,
    predicted label, the fold's fit and predict times (ms) and a probability per
    class. Finished folds are kept in a ``.partial`` file, so a killed run resumes.

    Parameters
    ----------
    problem_path : str
        Path to the problem folder; the dataset needs subject ID files and case tables.
    results_path : str
        Path to the results folder.
    dataset : str
        Name of the dataset.
    classifier : BaseClassifier
        Unfitted classifier, cloned for each fold. A ``train_groups`` parameter is set
        to None, as one subject cannot be split by subject.
    classifier_name : str or None, default=None
        Name for the results folder, the classifier's class name if None.
    subject_index : int, default=0
        Position of the subject among the sorted unique subject IDs.
    n_folds : int, default=10
        Number of contiguous folds.
    overwrite : bool, default=False
        Rerun even if the subject's file exists.
    """
    name = classifier_name or type(classifier).__name__
    folder = f"{results_path}/{name}/WithinSubject/{dataset}"
    out = f"{folder}/subject{subject_index}.csv"
    if os.path.exists(out) and not overwrite:
        warnings.warn(f"{out} already present, skipping.", stacklevel=1)
        return

    X_train, y_train, X_test, y_test, _ = load_experiment_data(
        problem_path, dataset, 0, False
    )
    n_train = len(y_train)
    y = np.concatenate((y_train, y_test), axis=None)
    if isinstance(X_train, np.ndarray) and isinstance(X_test, np.ndarray):
        X = np.concatenate((X_train, X_test), axis=0)
    else:
        X = list(X_train) + list(X_test)
    groups = load_subject_ids(problem_path, dataset)
    if groups is None:
        raise ValueError(f"{dataset} has no subject ID files")
    order = load_case_order(problem_path, dataset)
    if len(groups) != len(y) or len(order) != len(y):
        raise ValueError(
            f"{dataset}: {len(y)} cases, {len(groups)} subject IDs, {len(order)} "
            "case table rows"
        )
    subject, folds = subject_folds(groups, order, subject_index, n_folds)
    position = {int(i): p for p, i in enumerate(np.concatenate(folds))}
    classes = np.unique(y)

    if "train_groups" in classifier.get_params(deep=False):
        classifier.set_params(train_groups=None)

    os.makedirs(folder, exist_ok=True)
    partial = f"{out}.partial"
    header = ["fold", "source_split", "source_row", "subject_position", "true", "pred"]
    header += ["fit_ms", "predict_ms"] + [f"p_{c}" for c in classes]
    done = set()
    if os.path.exists(partial) and not overwrite:
        with open(partial, newline="") as file:
            rows = list(csv.reader(file))
        if rows and rows[0] == header:
            done = {int(r[0]) for r in rows[1:]}
        else:
            os.remove(partial)
    if not os.path.exists(partial) or overwrite:
        with open(partial, "w", newline="") as file:
            csv.writer(file).writerow(header)

    for fold, test_idx in enumerate(folds):
        if fold in done:
            continue
        train_idx = np.concatenate([f for k, f in enumerate(folds) if k != fold])
        estimator = clone(classifier)
        start = time.perf_counter()
        estimator.fit(_subset(X, train_idx), y[train_idx])
        fit_ms = int((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        fold_proba = estimator.predict_proba(_subset(X, test_idx))
        predict_ms = int((time.perf_counter() - start) * 1000)
        proba = np.zeros((len(test_idx), len(classes)))
        columns = np.searchsorted(classes, np.asarray(estimator.classes_))
        proba[:, columns] = fold_proba
        preds = classes[np.argmax(proba, axis=1)]
        with open(partial, "a", newline="") as file:
            writer = csv.writer(file)
            for i, pred, p in zip(test_idx, preds, proba):
                i = int(i)
                split, row = ("TRAIN", i) if i < n_train else ("TEST", i - n_train)
                writer.writerow(
                    [fold, split, row, position[i], y[i], pred, fit_ms, predict_ms]
                    + [f"{v:.6g}" for v in p]
                )

    with open(partial, newline="") as file:
        rows = list(csv.reader(file))
    settings = (
        f"# dataset={dataset}, classifier={name}, subject_index={subject_index}, "
        f"subject={subject}, n_cases={sum(len(f) for f in folds)}, n_folds={n_folds}, "
        "folds=contiguous in recording order (session, run, start), "
        f"generated={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
        f"params={classifier.get_params(deep=False)}"
    )
    with open(out, "w", newline="") as file:
        file.write(settings.replace("\n", " ") + "\n")
        csv.writer(file).writerows(
            [rows[0]] + sorted(rows[1:], key=lambda r: int(r[3]))
        )
    os.remove(partial)
