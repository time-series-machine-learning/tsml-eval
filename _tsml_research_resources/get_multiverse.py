from time import perf_counter

import numpy as np

from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import multiverse_core, multiverse2026

path = "/gpfs/home/ajb/Data/Multiverse"

datasets = multiverse2026

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


for problem in datasets:
    print(f"Loading {problem}...", flush=True)  # noqa: T201
    start = perf_counter()
    X, y, metadata = load_classification(
        problem,
        extract_path=path,
        return_metadata=True,
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

