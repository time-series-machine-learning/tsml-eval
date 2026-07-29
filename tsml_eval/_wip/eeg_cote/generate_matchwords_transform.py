"""Fit one MatchWords transform on TRAIN and write transformed TRAIN/TEST files."""

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
    _make_gmarv4_transformer,
    _metadata_to_builtin,
    _selector_metadata,
)

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - Windows development only
    resource = None

DEFAULT_INPUT_ROOT = Path("/iridisfs/home/ajb2u23/Data/EEG")
DEFAULT_OUTPUT_ROOT = Path("/iridisfs/home/ajb2u23/Data/EEGTransforms")
DEFAULT_PROBLEM = "MatchWords"
SUPPORTED_VARIANTS = (
    "TSelect",
    "GMARv3",
    "GMARv4-Arsenal",
    "GMARv4-DrCIF",
    "GMARv4-STC",
    "GMARv4-TDE",
)


def _source_file(source: Path, problem: str, split: str) -> Path:
    """Return the standard source file or the current duplicated-suffix file."""
    standard = source / f"{problem}_{split}.ts"
    duplicated_suffix = source / f"{problem}_{split}.ts.ts"
    if standard.is_file():
        return standard
    if duplicated_suffix.is_file():
        return duplicated_suffix
    raise FileNotFoundError(
        f"Expected {standard} or {duplicated_suffix}."
    )


def _make_transformer(variant: str, n_channels: int, random_state: int):
    """Construct the requested shared or component-specific transform."""
    if variant == "TSelect":
        return _make_channel_transformer(
            selector="TSelect",
            n_channels=n_channels,
            random_state=random_state,
            n_jobs=1,
        )
    if variant == "GMARv3":
        return _make_channel_transformer(
            selector="GuardedTemporalV3",
            n_channels=n_channels,
            random_state=random_state,
            n_jobs=1,
            proxy_component="HC2",
        )
    if variant.startswith("GMARv4-"):
        return _make_gmarv4_transformer(
            component=variant.removeprefix("GMARv4-"),
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(
        f"Unknown variant {variant!r}; expected one of {SUPPORTED_VARIANTS}."
    )


def _class_counts(y) -> dict[str, int]:
    """Return JSON-safe class frequencies."""
    labels, counts = np.unique(y, return_counts=True)
    return {
        str(label.item() if hasattr(label, "item") else label): int(count)
        for label, count in zip(labels, counts)
    }


def _candidate_records(transformer) -> list[dict]:
    """Return the compact GMAR candidate trace when available."""
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


def _json_safe(value):
    """Replace non-finite diagnostics before strict JSON serialisation."""
    value = _metadata_to_builtin(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _peak_rss_kib() -> int | None:
    """Return Linux peak resident memory, if the platform provides it."""
    if resource is None:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _reduction_summary(transformer) -> dict:
    """Return selected channels and reduction diagnostics without large indices."""
    metadata = {
        "transformer_class": type(transformer).__name__,
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
        metadata["reduction_candidates"] = candidates
    return _metadata_to_builtin(metadata)


def _complete_output(destination: Path, problem: str) -> bool:
    """Return whether both transformed splits and metadata are non-empty."""
    required = (
        destination / f"{problem}_TRAIN.ts",
        destination / f"{problem}_TEST.ts",
        destination / "transform_summary.json",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def transform_matchwords(
    *,
    input_root: Path,
    output_root: Path,
    problem: str,
    variant: str,
    random_state: int,
    overwrite: bool,
) -> str:
    """Fit one transform on TRAIN and persist aligned transformed splits."""
    source = input_root / problem
    train_file = _source_file(source, problem, "TRAIN")
    test_file = _source_file(source, problem, "TEST")
    destination = output_root / variant / problem
    if not overwrite and _complete_output(destination, problem):
        print(f"{variant}: complete output exists; skipping.")
        return "skipped"

    print(f"{variant}: loading {train_file}")
    load_start = time.perf_counter()
    X_train, y_train = load_from_ts_file(
        train_file,
        return_type="numpy3D",
    )
    print(f"{variant}: loading {test_file}")
    X_test, y_test = load_from_ts_file(
        test_file,
        return_type="numpy3D",
    )
    load_seconds = time.perf_counter() - load_start

    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            "TRAIN and TEST channel/time dimensions differ: "
            f"{X_train.shape} and {X_test.shape}."
        )
    if len(y_train) != X_train.shape[0] or len(y_test) != X_test.shape[0]:
        raise ValueError("The loaded cases and labels are not aligned.")

    transformer = _make_transformer(
        variant=variant,
        n_channels=X_train.shape[1],
        random_state=random_state,
    )
    print(
        f"{variant}: fitting transform on TRAIN {X_train.shape}; "
        f"TEST remains held out"
    )
    fit_start = time.perf_counter()
    if hasattr(transformer, "fit_resample"):
        transformed_train, transformed_y_train = transformer.fit_resample(
            X_train,
            y_train,
        )
    else:
        transformed_train = transformer.fit_transform(X_train, y_train)
        transformed_y_train = y_train
    fit_seconds = time.perf_counter() - fit_start

    if not np.array_equal(transformed_y_train, y_train):
        raise RuntimeError(
            "MatchWords transforms must retain all TRAIN labels in order."
        )

    test_start = time.perf_counter()
    transformed_test = transformer.transform(X_test)
    test_transform_seconds = time.perf_counter() - test_start
    if transformed_train.shape[1:] != transformed_test.shape[1:]:
        raise RuntimeError(
            "The learned TRAIN and TEST representations are incompatible: "
            f"{transformed_train.shape} and {transformed_test.shape}."
        )

    destination.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    with TemporaryDirectory(
        prefix=f".{variant.lower()}-",
        dir=destination,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        header = (
            f"MatchWords {variant} representation. The transform was fitted "
            "only on the original TRAIN participants and applied unchanged "
            "to the held-out TEST participants."
        )
        save_to_ts_file(
            transformed_train,
            transformed_y_train,
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
            "variant": variant,
            "random_state": random_state,
            "source": {
                "train": str(train_file),
                "test": str(test_file),
                "train_bytes": train_file.stat().st_size,
                "test_bytes": test_file.stat().st_size,
            },
            "participants": {
                "train": 9,
                "test": 9,
                "cases_per_participant_per_class": 20,
                "participant_disjoint_split_assumed": True,
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
            "transform": _reduction_summary(transformer),
        }
        temporary_summary = temporary / "transform_summary.json"
        temporary_summary.write_text(
            json.dumps(
                _json_safe(summary),
                indent=2,
                allow_nan=False,
            ),
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
        os.replace(
            temporary_summary,
            destination / "transform_summary.json",
        )
    write_seconds = time.perf_counter() - write_start

    print(
        f"{variant}: wrote TRAIN {transformed_train.shape}, "
        f"TEST {transformed_test.shape}"
    )
    print(
        f"{variant}: load={load_seconds / 60:.2f} min, "
        f"fit={fit_seconds / 60:.2f} min, "
        f"test transform={test_transform_seconds / 60:.2f} min, "
        f"write={write_seconds / 60:.2f} min"
    )
    return "written"


def main() -> None:
    """Generate one MatchWords representation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=SUPPORTED_VARIANTS,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    transform_matchwords(
        input_root=args.input_root,
        output_root=args.output_root,
        problem=args.problem,
        variant=args.variant,
        random_state=args.random_state,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
