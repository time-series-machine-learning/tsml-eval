import shutil
from pathlib import Path
from time import perf_counter

import numpy as np

from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import multiverse_core, multiverse2026

path = "/gpfs/home/ajb/Data/Multiverse"

datasets = multiverse2026


def _is_unequal_or_missing(train_file):
    equal_length = None
    missing = None

    with train_file.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            fields = line.strip().lower().split(maxsplit=1)
            if not fields:
                continue
            if fields[0] == "@data":
                break
            if len(fields) == 2:
                if fields[0] == "@equallength":
                    equal_length = fields[1] == "true"
                elif fields[0] == "@missing":
                    missing = fields[1] == "true"

    return equal_length is False or missing is True


def delete_downloaded_unequal_or_missing(extract_path):
    root = Path(extract_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".unequal_missing_cleanup_complete"

    if marker.exists():
        print(  # noqa: T201
            "One-off unequal/missing cleanup already completed.", flush=True
        )
        return

    for problem_dir in root.iterdir():
        if not problem_dir.is_dir() or problem_dir.is_symlink():
            continue

        train_file = problem_dir / f"{problem_dir.name}_TRAIN.ts"
        if train_file.is_file() and _is_unequal_or_missing(train_file):
            print(  # noqa: T201
                f"Deleting {problem_dir} for clean redownload...", flush=True
            )
            shutil.rmtree(problem_dir)

    marker.touch()


def _series_characteristics(X):
    if isinstance(X, np.ndarray):
        if X.ndim == 3:
            return X.shape[1], str(X.shape[2])
        if X.ndim == 2:
            return 1, str(X.shape[1])

    n_channels = X[0].shape[0]
    lengths = [case.shape[-1] for case in X]
    length = (
        str(lengths[0])
        if min(lengths) == max(lengths)
        else f"{min(lengths)}-{max(lengths)}"
    )
    return n_channels, length


delete_downloaded_unequal_or_missing(path)

for problem in datasets:
    problem_path = Path(path) / problem
    if problem_path.exists():
        print(  # noqa: T201
            f"Skipping {problem}: already exists at {problem_path}", flush=True
        )
        continue

    print(f"Loading {problem}...", flush=True)  # noqa: T201
    start = perf_counter()
    X, y, metadata = load_classification(
        problem,
        extract_path=path,
        return_metadata=True,
        load_equal_length=True,
        load_no_missing=True,
    )
    elapsed = perf_counter() - start

    n_channels, series_length = _series_characteristics(X)
    print(  # noqa: T201
        f"Loaded {problem} in {elapsed:.2f} seconds: "
        f"cases={len(y)}, classes={len(np.unique(y))}, "
        f"channels={n_channels}, series_length={series_length}, "
        f"equal_length={metadata['equallength']}, missing={metadata['missing']}",
        flush=True,
    )
