"""Recover expensive EEG pipeline train files using parallel CV folds.

The standard experiment runner generates pipeline train predictions with a
serial ``cross_val_predict`` call. Some of the large EEG pipelines therefore
exceed Iridis' 60-hour batch limit even though each individual CV fold fits
comfortably inside it. This utility runs one deterministic stratified fold at
a time and combines the ten fold outputs into a normal tsml train result.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from tsml_eval.experiments._get_classifier import get_classifier_by_name
from tsml_eval.utils.datasets import load_experiment_data
from tsml_eval.utils.resampling import stratified_resample_data
from tsml_eval.utils.results_writing import write_classification_results


def _load_training_data(data_path, dataset, resample_id):
    X_train, y_train, X_test, y_test, resample = load_experiment_data(
        data_path,
        dataset,
        resample_id,
        predefined_resample=False,
    )
    if resample:
        X_train, y_train, _, _ = stratified_resample_data(
            X_train,
            y_train,
            X_test,
            y_test,
            random_state=resample_id,
        )

    encoder = LabelEncoder()
    return X_train, encoder.fit_transform(y_train)


def _take_cases(X, indices):
    if isinstance(X, np.ndarray):
        return X[indices]
    return [X[index] for index in indices]


def _fold_indices(y, n_splits, fold):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    splits = list(cv.split(np.zeros(len(y), dtype=np.int8), y))
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must be in 0..{len(splits) - 1}, found {fold}")
    return splits[fold]


def _valid_partial(path, classifier, dataset, resample_id, fold, n_splits):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as result:
            return (
                str(result["classifier"].item()) == classifier
                and str(result["dataset"].item()) == dataset
                and int(result["resample_id"].item()) == resample_id
                and int(result["fold"].item()) == fold
                and int(result["n_splits"].item()) == n_splits
                and len(result["test_indices"]) == len(result["y_true"])
                and result["probabilities"].shape[0] == len(result["y_true"])
            )
    except (OSError, KeyError, ValueError):
        return False


def run_fold(args):
    partial_path = Path(args.partial_dir) / f"fold{args.fold}.npz"
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    if _valid_partial(
        partial_path,
        args.classifier,
        args.dataset,
        args.resample_id,
        args.fold,
        args.n_splits,
    ):
        print(f"Fold {args.fold} already complete: {partial_path}")
        return

    X, y = _load_training_data(args.data_path, args.dataset, args.resample_id)
    train_indices, test_indices = _fold_indices(y, args.n_splits, args.fold)

    classifier = get_classifier_by_name(
        args.classifier,
        random_state=args.resample_id,
        n_jobs=1,
    )

    start = time.time_ns()
    classifier.fit(_take_cases(X, train_indices), y[train_indices])
    probabilities = classifier.predict_proba(_take_cases(X, test_indices))
    elapsed_millis = (time.time_ns() - start) // 1_000_000

    expected_classes = np.arange(len(np.unique(y)))
    if not np.array_equal(classifier.classes_, expected_classes):
        raise RuntimeError(
            "Fold classifier probability columns are not in global class order: "
            f"expected {expected_classes}, found {classifier.classes_}."
        )
    if probabilities.shape != (len(test_indices), len(expected_classes)):
        raise RuntimeError(
            "Unexpected probability shape: "
            f"expected {(len(test_indices), len(expected_classes))}, "
            f"found {probabilities.shape}."
        )

    temporary_path = partial_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            classifier=np.asarray(args.classifier),
            dataset=np.asarray(args.dataset),
            resample_id=np.asarray(args.resample_id),
            fold=np.asarray(args.fold),
            n_splits=np.asarray(args.n_splits),
            test_indices=np.asarray(test_indices),
            y_true=np.asarray(y[test_indices]),
            probabilities=np.asarray(probabilities),
            elapsed_millis=np.asarray(elapsed_millis),
        )
    os.replace(temporary_path, partial_path)
    print(
        f"Completed {args.classifier}/{args.dataset} fold {args.fold}: "
        f"{elapsed_millis / 3_600_000:.2f} hours"
    )


def combine_folds(args):
    result_path = (
        Path(args.results_path)
        / args.classifier
        / "Predictions"
        / args.dataset
        / f"trainResample{args.resample_id}.csv"
    )
    if result_path.is_file() and result_path.stat().st_size > 0 and not args.overwrite:
        print(f"Train result already exists: {result_path}")
        return

    partial_dir = Path(args.partial_dir)
    partials = [partial_dir / f"fold{fold}.npz" for fold in range(args.n_splits)]
    missing = [str(path) for path in partials if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing fold results:\n  " + "\n  ".join(missing))

    fold_data = []
    n_cases = 0
    n_classes = None
    elapsed_millis = 0
    for fold, path in enumerate(partials):
        if not _valid_partial(
            path,
            args.classifier,
            args.dataset,
            args.resample_id,
            fold,
            args.n_splits,
        ):
            raise ValueError(f"Invalid fold result: {path}")
        with np.load(path, allow_pickle=False) as result:
            indices = result["test_indices"].copy()
            labels = result["y_true"].copy()
            probabilities = result["probabilities"].copy()
            fold_data.append((indices, labels, probabilities))
            n_cases += len(indices)
            elapsed_millis += int(result["elapsed_millis"].item())
            if n_classes is None:
                n_classes = probabilities.shape[1]
            elif n_classes != probabilities.shape[1]:
                raise ValueError("CV folds contain different numbers of classes.")

    all_indices = np.concatenate([result[0] for result in fold_data])
    if not np.array_equal(np.sort(all_indices), np.arange(n_cases)):
        raise ValueError("CV fold indices do not partition the complete train set.")

    labels = np.empty(n_cases, dtype=fold_data[0][1].dtype)
    probabilities = np.empty((n_cases, n_classes), dtype=float)
    for indices, fold_labels, fold_probabilities in fold_data:
        labels[indices] = fold_labels
        probabilities[indices] = fold_probabilities

    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(labels, predictions)

    test_path = result_path.with_name(f"testResample{args.resample_id}.csv")
    parameter_info = "Parallel CV recovery; estimator parameters unavailable."
    if test_path.is_file():
        with test_path.open(encoding="utf-8") as test_file:
            test_file.readline()
            parameter_info = test_file.readline().rstrip("\n\r")

    write_classification_results(
        predictions,
        probabilities,
        labels,
        args.classifier,
        args.dataset,
        args.results_path,
        full_path=False,
        first_line_classifier_name=(
            f"{args.classifier} (ChannelSelectionClassifierPipeline)"
        ),
        split="TRAIN",
        resample_id=args.resample_id,
        time_unit="MILLISECONDS",
        first_line_comment=(
            f"Recovered from {args.n_splits} deterministic stratified CV folds "
            "run independently on Iridis."
        ),
        parameter_info=parameter_info,
        accuracy=accuracy,
        fit_time=-1,
        predict_time=-1,
        benchmark_time=-1,
        memory_usage=-1,
        n_classes=n_classes,
        train_estimate_method=f"{args.n_splits}F-CV-PARALLEL",
        train_estimate_time=elapsed_millis,
        fit_and_estimate_time=-1,
    )
    print(
        f"Wrote {result_path}: accuracy={accuracy:.6f}, "
        f"aggregate fold time={elapsed_millis / 3_600_000:.2f} hours"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("data_path")
    common.add_argument("results_path")
    common.add_argument("classifier")
    common.add_argument("dataset")
    common.add_argument("partial_dir")
    common.add_argument("--resample-id", type=int, default=0)
    common.add_argument("--n-splits", type=int, default=10)

    fold_parser = subparsers.add_parser("fold", parents=[common])
    fold_parser.add_argument("fold", type=int)
    fold_parser.set_defaults(function=run_fold)

    combine_parser = subparsers.add_parser("combine", parents=[common])
    combine_parser.add_argument("--overwrite", action="store_true")
    combine_parser.set_defaults(function=combine_folds)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    parsed_args.function(parsed_args)
