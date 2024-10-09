"""Runtime and memory usage benchmarking."""

from dataclasses import dataclass
from math import floor

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state

__author__ = ["GuiArcencio"]
__all__ = [
    "BenchmarkResult",
    "ComparisonResult",
    "benchmark_estimator",
    "compare_estimators",
]

from tsml_eval.utils.memory_recorder import record_max_memory


@dataclass
class BenchmarkResult:
    """Aggregates runtimes (milliseconds) and memory usage (bytes)."""

    fit_runtime: float
    predict_runtime: float
    fit_memory_usage: int
    predict_memory_usage: int


@dataclass
class ComparisonResult:
    """Aggregates results from comparing estimators' benchmarks."""

    total_size: int
    train_size: int
    test_size: int

    split_seed: int

    benchmark_1: BenchmarkResult
    benchmark_2: BenchmarkResult


def benchmark_estimator(
    estimator, X, y=None, train_size=None, test_size=None, random_state=None
):
    """Benchmark `estimator`'s runtime and memory usage on `X` and `y`.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator to be benchmarked.
    X : np.ndarray or pd.DataFrame
        Input data to train and test the estimator on.
    y : array-like of shape [n_instances], default=None
        Response data in case of supervised tasks.
    train_size : float or int, default=None
        Size of train split.
    test_size : float or int, default=None
        Size of test split.
    random_state : int, np.random.RandomState or None, default=None
        Randomizer seed or RandomState.

    Returns
    -------
    result : BenchmarkResult
        Benchmark aggregated results.
    """
    estimator = clone(estimator)
    rng = check_random_state(random_state)
    if y is None:
        # Suppose clustering
        X_train, X_test = train_test_split(
            X, train_size=train_size, test_size=test_size, random_state=rng
        )
        y_train = None
    else:
        X_train, X_test, y_train, _ = train_test_split(
            X,
            y,
            train_size=train_size,
            test_size=test_size,
            random_state=rng,
        )

    memory_fit, runtime_fit = record_max_memory(
        estimator.fit, args=(X_train, y_train), return_func_time=True
    )
    memory_predict, runtime_predict = record_max_memory(
        estimator.predict, args=(X_test,), return_func_time=True
    )

    return BenchmarkResult(
        fit_runtime=runtime_fit,
        predict_runtime=runtime_predict,
        fit_memory_usage=memory_fit,
        predict_memory_usage=memory_predict,
    )


def compare_estimators(
    estimator_1,
    estimator_2,
    X,
    y=None,
    varying="total",
    sizes=None,
    fixed_size=None,
    random_state=None,
):
    """Perform a sequence of benchmarks for two estimators.

    The total number of benchmarks will always be `len(sizes)`.

    If `varying` is `"total"`, `sizes` will be interpreted as
    the total size (train + test) of each run, with the test proportion
    of the split set to `fixed_size` (between 0 and 1).

    If `varying` is `"train"`, `sizes` will be interpreted as the
    training set sizes of each run, with the test size set to `fixed_size` (integer).

    If `varying` is `"test"`, `sizes` will be interpreted as the
    testing set sizes of each run, with the train size set to `fixed_size` (integer).

    Parameters
    ----------
    estimator_1 : BaseEstimator
        First estimator to be benchmarked.
    estimator_2 : BaseEstimator
        Second estimator to be benchmarked.
    X : np.ndarray or pd.DataFrame
        Input data to train and test the estimators on.
    y : array-like of shape [n_instances], default=None
        Response data in case of supervised tasks.
    varying : {'total', 'train' or 'test'}, default='total'
        Whether to iterate through total sizes, train sizes or test sizes.
    sizes : array-like of integers, default=None
        Sequence of sizes to benchmark on. Will default to `X.shape[0]` if
        `varying` is 'total'.
    fixed_size : float or int, default=None
        Testing set proportion, fixed train size or fixed test size,
        depending on `varying`.
    random_state : int, np.random.RandomState or None, default=None
        Randomizer seed or RandomState.

    Returns
    -------
    results : list of ComparisonResult
        Aggregated results for each size.
    """
    rng = check_random_state(random_state)

    if varying == "total":
        if fixed_size is None:
            fixed_size = 0.25
        elif fixed_size >= 1.0 or fixed_size <= 0:
            raise ValueError(
                f"Invalid fixed size: {fixed_size}. If 'varying' is 'total',"
                + " 'fixed_size' must be between 0 and 1."
            )

        if sizes is None:
            sizes = X.shape[0]

        sizes = np.array(sizes, ndmin=1)
        results = []
        for size in sizes:
            split_seed = rng.randint(2**32)
            test_size = floor(fixed_size * size)
            train_size = size - test_size

            first_result = benchmark_estimator(
                estimator_1, X, y, train_size, test_size, split_seed
            )
            second_result = benchmark_estimator(
                estimator_2, X, y, train_size, test_size, split_seed
            )

            results.append(
                ComparisonResult(
                    total_size=size,
                    train_size=train_size,
                    test_size=test_size,
                    split_seed=split_seed,
                    benchmark_1=first_result,
                    benchmark_2=second_result,
                )
            )

        return results
    elif varying == "train" or varying == "test":
        if sizes is None or fixed_size is None:
            raise ValueError(
                "If varying train or test sizes, neither 'sizes' nor"
                + " 'fixed_size' can be None."
            )

        sizes = np.array(sizes, ndmin=1)
        results = []
        for size in sizes:
            split_seed = rng.randint(2**32)
            if varying == "train":
                train_size = size
                test_size = fixed_size
            else:
                test_size = size
                train_size = fixed_size

            first_result = benchmark_estimator(
                estimator_1, X, y, train_size, test_size, split_seed
            )
            second_result = benchmark_estimator(
                estimator_2, X, y, train_size, test_size, split_seed
            )

            results.append(
                ComparisonResult(
                    total_size=train_size + test_size,
                    train_size=train_size,
                    test_size=test_size,
                    split_seed=split_seed,
                    benchmark_1=first_result,
                    benchmark_2=second_result,
                )
            )

        return results
    else:
        raise ValueError(
            f"Invalid varying method: {varying}. Allowed values"
            + " are {'total', 'train', 'test'}."
        )
