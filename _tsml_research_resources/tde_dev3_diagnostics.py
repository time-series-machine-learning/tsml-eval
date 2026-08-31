"""Small diagnostics for TDE Dev3 per-dimension similarity fusion.

This is deliberately not an archive benchmark. It checks the proposed similarity on
one exact sparse-bag example and one synthetic multivariate train/test problem.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import accuracy_score

from tsml_eval._wip.tde_dev._tde_dev3 import (
    IndividualTDE_Dev3,
    normalised_dimension_similarity,
)


def _representation(dimensions):
    key1_parts = []
    key2_parts = []
    count_parts = []
    offsets = []
    starts = []
    masses = []
    start = 0
    for cases in dimensions:
        starts.append(start)
        dim_offsets = [0]
        dim_masses = []
        for words, counts in cases:
            key1_parts.extend([0] * len(words))
            key2_parts.extend(words)
            count_parts.extend(counts)
            dim_offsets.append(dim_offsets[-1] + len(words))
            dim_masses.append(sum(counts))
        offsets.append(dim_offsets)
        masses.append(dim_masses)
        start += dim_offsets[-1]
    return (
        np.asarray(key1_parts, dtype=np.uint64),
        np.asarray(key2_parts, dtype=np.uint64),
        np.asarray(count_parts, dtype=np.uint32),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
        np.asarray(masses, dtype=np.float64),
    )


def unequal_bag_mass_diagnostic():
    """Return raw and normalised similarities for an exact two-channel example."""
    query = _representation([[([1], [2])], [([9], [100])]])
    informative_match = _representation([[([1], [2])], [([9], [1])]])
    large_mass_match = _representation([[([2], [2])], [([9], [100])]])
    weights = np.asarray([0.5, 0.5])
    return {
        "raw_informative": 3,
        "raw_large_mass": 100,
        "normalised_informative": normalised_dimension_similarity(
            *query, 0, *informative_match, 0, weights
        ),
        "normalised_large_mass": normalised_dimension_similarity(
            *query, 0, *large_mass_match, 0, weights
        ),
    }


def _make_strong_weak_problem(n_cases, seed):
    """Create smooth strong/moderate channels plus three high-mass noise channels."""
    rng = np.random.RandomState(seed)
    y = np.arange(n_cases, dtype=np.int64) % 2
    rng.shuffle(y)
    length = 60
    t = np.linspace(0, 2 * np.pi, length)
    X = np.empty((n_cases, 5, length), dtype=np.float64)
    for i, label in enumerate(y):
        phase = 0 if label == 0 else np.pi / 2
        X[i, 0] = np.sin(t + phase) + rng.normal(0, 0.08, length)
        X[i, 1] = 0.45 * np.sin(t + phase) + rng.normal(0, 0.45, length)
        X[i, 2:] = rng.normal(0, 1, (3, length))
    return X, y


def strong_weak_diagnostic(seed=17):
    """Compare all three fusion choices with identical IndividualTDE settings."""
    X_train, y_train = _make_strong_weak_problem(40, seed)
    X_test, y_test = _make_strong_weak_problem(100, seed + 1)
    common = {
        "window_size": 12,
        "word_length": 8,
        "norm": True,
        "levels": 1,
        "igb": True,
        "bigrams": False,
        "dim_threshold": 0.0,
        "max_dims": 5,
        "random_state": seed,
    }
    modes = {
        "merged": ("merged", "uniform"),
        "normalised_uniform": ("normalised", "uniform"),
        "normalised_accuracy": ("normalised", "accuracy"),
    }
    results = {}
    for name, (similarity, weighting) in modes.items():
        parameters = {
            **common,
            "multivariate_similarity": similarity,
            "dimension_weighting": weighting,
        }
        # Compile/warm the corresponding Numba path before measuring it.
        warm_estimator = IndividualTDE_Dev3(**parameters).fit(X_train, y_train)
        warm_estimator.predict(X_test[:2])

        estimator = IndividualTDE_Dev3(**parameters)
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predictions = estimator.predict(X_test)
        predict_seconds = time.perf_counter() - start
        representation_bytes = sum(
            array.nbytes
            for array in estimator._transformed_data
            if isinstance(array, np.ndarray)
        )
        results[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "representation_bytes": representation_bytes,
            "dimension_scores": estimator._dimension_scores_.tolist(),
            "dimension_weights": estimator._dimension_weights_.tolist(),
            "mean_bag_masses": (
                estimator._transformed_data[-1].mean(axis=1).tolist()
                if similarity == "normalised"
                else None
            ),
        }
    return results


def main():
    bag_result = unequal_bag_mass_diagnostic()
    print("Unequal bag-mass diagnostic")
    for name, value in bag_result.items():
        print(f"  {name}: {value:.6f}")

    print("\nStrong + weak dimensions diagnostic")
    for name, result in strong_weak_diagnostic().items():
        print(
            f"  {name}: accuracy={result['accuracy']:.3f}, "
            f"fit={result['fit_seconds']:.3f}s, "
            f"predict={result['predict_seconds']:.3f}s, "
            f"representation={result['representation_bytes']} bytes"
        )
        print(f"    scores={np.round(result['dimension_scores'], 3).tolist()}")
        print(f"    weights={np.round(result['dimension_weights'], 3).tolist()}")
        if result["mean_bag_masses"] is not None:
            print(
                "    mean_bag_masses="
                f"{np.round(result['mean_bag_masses'], 3).tolist()}"
            )


if __name__ == "__main__":
    main()
