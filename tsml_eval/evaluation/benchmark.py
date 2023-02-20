# -*- coding: utf-8 -*-
"""Runtime and memory usage benchmarking."""

from dataclasses import dataclass
from time import perf_counter

import psutil
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state

__author__ = ["GuiArcencio"]


@dataclass
class BenchmarkResult:
    """Aggregates runtimes (seconds) and memory usage (bytes)."""

    total_runtime: float
    fit_runtime: float
    predict_runtime: float

    total_memory_usage: int
    fit_memory_usage: int
    predict_memory_usage: int


def benchmark_estimator(
    estimator, X, y=None, train_size=None, test_size=None, random_state=None
):
    """Benchmarks `estimator`'s runtime and memory usage on `X` and `y`."""
    rng = check_random_state(random_state)
    if y is None:
        # Suppose clustering
        X_train, X_test = train_test_split(
            X, train_size=train_size, test_size=test_size, random_state=rng
        )
        y_train = None
    else:
        X_train, X_test, y_train, _ = train_test_split(
            X, y, train_size=train_size, test_size=test_size, random_state=rng
        )

    runtime_fit, memory_fit, _ = _benchmark_function_wrapper(
        estimator.fit, args=[X_train, y_train], kwargs={}
    )
    runtime_predict, memory_predict, _ = _benchmark_function_wrapper(
        estimator.predict, args=[X_test], kwargs={}
    )

    return BenchmarkResult(
        fit_runtime=runtime_fit,
        predict_runtime=runtime_predict,
        total_runtime=runtime_fit + runtime_predict,
        fit_memory_usage=memory_fit,
        predict_memory_usage=memory_predict,
        total_memory_usage=memory_fit + memory_predict,
    )


def _benchmark_function_wrapper(func, args, kwargs):
    process = psutil.Process()

    mem_before = process.memory_info().vms
    clock_start = perf_counter()
    func_output = func(*args, **kwargs)
    clock_end = perf_counter()
    mem_after = process.memory_info().vms

    return clock_end - clock_start, mem_after - mem_before, func_output
