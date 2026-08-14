"""Fit and persist one transform for a large EEG archive problem.

This is a recovery utility for experiments where fitting a transform and the
downstream classifier in one process exceeds a cluster wall-time. The transform
is fitted on TRAIN only and applied unchanged to TEST. Both transformed splits
and a timing/provenance summary are written below ``Results/Transforms``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from aeon.datasets import load_from_ts_file, save_to_ts_file

from tsml_eval.experiments._channel_selection_hc2 import (
    _make_channel_transformer,
    _metadata_to_builtin,
    _selector_metadata,
)

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - Windows development only
    resource = None


SUPPORTED_TRANSFORMS = ("GEAR-Auto",)


def _source_file(source: Path, problem: str, split: str) -> Path:
    """Find a standard ts file, tolerating the historical double suffix."""
    standard = source / f"{problem}_{split}.ts"
    duplicated_suffix = source / f"{problem}_{split}.ts.ts"
    if standard.is_file():
        return standard
    if duplicated_suffix.is_file():
        return duplicated_suffix
    raise FileNotFoundError(f"Expected {standard} or {duplicated_suffix}.")


def _make_transformer(name: str, n_channels: int, random_state: int):
    """Construct the archive transform without its downstream classifier."""
    if name == "GEAR-Auto":
        return _make_channel_transformer(
            selector="GEARAuto",
            n_channels=n_channels,
            random_state=random_state,
            n_jobs=1,
            proxy_component="HC2",
        )
    raise ValueError(f"Unknown transform {name!r}; expected {SUPPORTED_TRANSFORMS}.")


def _class_counts(y) -> dict[str, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {
        str(label.item() if hasattr(label, "item") else label): int(count)
        for label, count in zip(labels, counts)
    }


def _json_safe(value):
    value = _metadata_to_builtin(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _peak_rss_kib() -> int | None:
    if resource is None:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _candidate_records(transformer) -> list[dict]:
    results = getattr(transformer, "candidate_results_", None)
    if results is None:
        return []
    records = results.to_dict("records")
    for record in records:
        if "fit_time" in record:
            record["fit_time_seconds"] = record.pop("fit_time")
        if "predict_time" in record:
            record["predict_time_seconds"] = record.pop("predict_time")
    return _metadata_to_builtin(records)


def _transform_metadata(transformer) -> dict:
    metadata = {
        "class": type(transformer).__name__,
        "selector": _selector_metadata(transformer),
    }
    if hasattr(transformer, "get_reduction_summary"):
        summary = transformer.get_reduction_summary()
        metadata["reduction_summary"] = {
            key: value
            for key, value in summary.items()
            if key not in {"case_indices", "time_indices"}
        }
    candidates = _candidate_records(transformer)
    if candidates:
        metadata["candidates"] = candidates
    return _metadata_to_builtin(metadata)


def _complete(destination: Path, problem: str) -> bool:
    required = (
        destination / f"{problem}_TRAIN.ts",
        destination / f"{problem}_TEST.ts",
        destination / "transform_summary.json",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def generate_transform(
    *,
    input_root: Path,
    output_root: Path,
    problem: str,
    transform_name: str,
    random_state: int,
    overwrite: bool,
) -> str:
    """Fit on TRAIN and atomically persist transformed TRAIN and TEST."""
    source = input_root / problem
    train_file = _source_file(source, problem, "TRAIN")
    test_file = _source_file(source, problem, "TEST")
    destination = output_root / transform_name / problem

    if not overwrite and _complete(destination, problem):
        print(f"{transform_name}/{problem}: complete transform exists; skipping.")
        return "skipped"

    load_start = time.perf_counter()
    X_train, y_train = load_from_ts_file(train_file, return_type="numpy3D")
    X_test, y_test = load_from_ts_file(test_file, return_type="numpy3D")
    load_seconds = time.perf_counter() - load_start

    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            "TRAIN and TEST channel/time dimensions differ: "
            f"{X_train.shape} and {X_test.shape}."
        )

    transformer = _make_transformer(
        transform_name,
        n_channels=X_train.shape[1],
        random_state=random_state,
    )
    print(
        f"{transform_name}/{problem}: fitting on TRAIN {X_train.shape}; "
        f"TEST {X_test.shape} remains held out"
    )
    fit_start = time.perf_counter()
    if hasattr(transformer, "fit_resample"):
        transformed_train, transformed_y_train = transformer.fit_resample(
            X_train, y_train
        )
    else:
        transformed_train = transformer.fit_transform(X_train, y_train)
        transformed_y_train = y_train
    fit_seconds = time.perf_counter() - fit_start

    if not np.array_equal(transformed_y_train, y_train):
        raise RuntimeError(
            "This recovery stage requires all TRAIN labels to remain aligned."
        )

    transform_start = time.perf_counter()
    transformed_test = transformer.transform(X_test)
    test_transform_seconds = time.perf_counter() - transform_start

    if transformed_train.shape[0] != X_train.shape[0]:
        raise RuntimeError("The transform unexpectedly changed the TRAIN case count.")
    if transformed_test.shape[0] != X_test.shape[0]:
        raise RuntimeError("The transform unexpectedly changed the TEST case count.")
    if transformed_train.shape[1:] != transformed_test.shape[1:]:
        raise RuntimeError(
            "Transformed TRAIN and TEST dimensions differ: "
            f"{transformed_train.shape} and {transformed_test.shape}."
        )

    destination.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    with TemporaryDirectory(prefix=".transform-", dir=destination) as temporary_dir:
        temporary = Path(temporary_dir)
        header = (
            f"{problem} {transform_name} representation. Transform fitted on "
            "TRAIN only and applied unchanged to TEST."
        )
        save_to_ts_file(
            transformed_train,
            y_train,
            label_type="classification",
            path=temporary,
            problem_name=problem,
            file_suffix="_TRAIN",
            header=header,
        )
        save_to_ts_file(
            transformed_test,
            y_test,
            label_type="classification",
            path=temporary,
            problem_name=problem,
            file_suffix="_TEST",
            header=header,
        )
        split_write_seconds = time.perf_counter() - write_start

        summary = {
            "problem": problem,
            "transform_name": transform_name,
            "random_state": random_state,
            "source": {
                "train": str(train_file),
                "test": str(test_file),
                "train_bytes": train_file.stat().st_size,
                "test_bytes": test_file.stat().st_size,
            },
            "class_counts": {
                "train": _class_counts(y_train),
                "test": _class_counts(y_test),
            },
            "shapes": {
                "train_input": list(X_train.shape),
                "test_input": list(X_test.shape),
                "train_output": list(transformed_train.shape),
                "test_output": list(transformed_test.shape),
            },
            "timings_seconds": {
                "load": load_seconds,
                "fit_transform_train": fit_seconds,
                "transform_test": test_transform_seconds,
                "write_splits": split_write_seconds,
            },
            "peak_rss_kib": _peak_rss_kib(),
            "transform": _transform_metadata(transformer),
        }
        summary_file = temporary / "transform_summary.json"
        summary_file.write_text(
            json.dumps(_json_safe(summary), indent=2, allow_nan=False),
            encoding="utf-8",
        )

        os.replace(
            temporary / f"{problem}_TRAIN.ts",
            destination / f"{problem}_TRAIN.ts",
        )
        os.replace(
            temporary / f"{problem}_TEST.ts",
            destination / f"{problem}_TEST.ts",
        )
        os.replace(summary_file, destination / "transform_summary.json")

    print(
        f"{transform_name}/{problem}: wrote TRAIN {transformed_train.shape}, "
        f"TEST {transformed_test.shape}; fit={fit_seconds / 3600:.2f} h, "
        f"TEST transform={test_transform_seconds / 60:.2f} min"
    )
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--transform", required=True, choices=SUPPORTED_TRANSFORMS)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generate_transform(
        input_root=args.input_root,
        output_root=args.output_root,
        problem=args.problem,
        transform_name=args.transform,
        random_state=args.random_state,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
