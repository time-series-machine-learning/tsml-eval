"""Utility functions for experiments."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "assign_gpu",
    "timing_benchmark",
    "estimator_attributes_to_file",
]

import os
import time
from collections.abc import Sequence

import gpustat
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


def _results_present(path, estimator, dataset, resample_id=None, split="TEST"):
    """Check if results are present already."""
    resample_str = "Results" if resample_id is None else f"Resample{resample_id}"
    path = f"{path}/{estimator}/Predictions/{dataset}/"

    if split == "BOTH":
        full_path = f"{path}test{resample_str}.csv"
        full_path2 = f"{path}train{resample_str}.csv"

        if os.path.exists(full_path) and os.path.exists(full_path2):
            return True
    else:
        if split is None or split == "" or split == "NONE":
            full_path = f"{path}{resample_str.lower()}.csv"
        elif split == "TEST":
            full_path = f"{path}test{resample_str}.csv"
        elif split == "TRAIN":
            full_path = f"{path}train{resample_str}.csv"
        else:
            raise ValueError(f"Unknown split value: {split}")

        if os.path.exists(full_path):
            return True

    return False


def _check_existing_results(
    results_path,
    estimator_name,
    dataset,
    resample_id,
    overwrite,
    build_test_file,
    build_train_file,
):
    """Check if results are present already and if they should be overwritten."""
    if not overwrite:
        resample_str = "Result" if resample_id is None else f"Resample{resample_id}"

        if build_test_file:
            full_path = (
                f"{results_path}/{estimator_name}/Predictions/{dataset}/"
                f"/test{resample_str}.csv"
            )

            if os.path.exists(full_path):
                build_test_file = False

        if build_train_file:
            full_path = (
                f"{results_path}/{estimator_name}/Predictions/{dataset}/"
                f"/train{resample_str}.csv"
            )

            if os.path.exists(full_path):
                build_train_file = False

    return build_test_file, build_train_file


def assign_gpu(set_environ=False):  # pragma: no cover
    """Assign a GPU to the current process.

    Looks at the available Nvidia GPUs and assigns the GPU with the lowest used memory.

    Parameters
    ----------
    set_environ : bool
        Set the CUDA_DEVICE_ORDER environment variable to "PCI_BUS_ID" anf the
        CUDA_VISIBLE_DEVICES environment variable to the assigned GPU.

    Returns
    -------
    gpu : int
        The GPU assigned to the current process.
    """
    stats = gpustat.GPUStatCollection.new_query()
    pairs = [
        [
            gpu.entry["index"],
            float(gpu.entry["memory.used"]) / float(gpu.entry["memory.total"]),
        ]
        for gpu in stats
    ]

    gpu = min(pairs, key=lambda x: x[1])[0]

    if set_environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    return gpu


def timing_benchmark(num_arrays=1000, array_size=20000, random_state=None):
    """
    Measures the time taken to sort a given number of numpy arrays of a specified size.

    Returns the time taken in milliseconds.

    Parameters
    ----------
    num_arrays: int, default=1000
        Number of arrays to generate and sort.
    array_size: int, default=20000
        Size of each numpy array to be sorted.
    random_state: int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    time_taken: int
        Time taken to sort the arrays in milliseconds.
    """
    if random_state is None:
        random_state = check_random_state(0)
    elif isinstance(random_state, (int, np.random.RandomState)):
        random_state = check_random_state(random_state)
    else:
        raise ValueError("random_state must be an int, RandomState instance or None")

    total_time = 0
    for _ in range(num_arrays):
        array = random_state.rand(array_size)
        start_time = time.time()
        np.sort(array)
        end_time = time.time()
        total_time += end_time - start_time

    return int(round(total_time * 1000))


def estimator_attributes_to_file(
    estimator, dir_path, estimator_name=None, max_depth=np.inf, max_list_shape=np.inf
):
    """Write the attributes of an estimator to file(s).

    Write the attributes of an estimator to file at a given directory. The function
    will recursively write the attributes of any estimators or non-string sequences
    containing estimators found in the attributes of the input estimator to spearate
    files.

    Parameters
    ----------
    estimator : estimator instance
        The estimator to write the attributes of.
    dir_path : str
        The directory to write the attribute files to.
    estimator_name : str or None, default=None
        The name of the estimator. If None, the name of the estimator class will be
        used.
    max_depth : int, default=np.inf
        The maximum depth to go when recursively writing attributes of estimators.
    max_list_shape : int, default=np.inf
        The maximum shape of a list to write when recursively writing attributes of
        contained estimators. i.e. for 0, no estimators contained in lists will be
        written, for 1, only estimators contained in 1-dimensional lists or the top
        level of a list will be written.
    """
    estimator_name = (
        estimator.__class__.__name__ if estimator_name is None else estimator_name
    )
    _write_estimator_attributes_recursive(
        estimator, dir_path, estimator_name, 0, max_depth, max_list_shape
    )


def _write_estimator_attributes_recursive(
    estimator, dir_path, file_name, depth, max_depth, max_list_shape
):
    if depth > max_depth:
        return

    path = f"{dir_path}/{file_name}.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        for attr in estimator.__dict__:
            value = getattr(estimator, attr)
            file.write(f"{attr}: {value}\n")

            if depth + 1 <= max_depth:
                if isinstance(value, BaseEstimator):
                    new_dir_path = f"{dir_path}/{attr}/"
                    file.write(f"    See {new_dir_path}{attr}.txt for more details\n")
                    _write_estimator_attributes_recursive(
                        value, new_dir_path, attr, depth + 1, max_depth, max_list_shape
                    )
                elif _is_non_string_sequence(value):
                    _write_list_attributes_recursive(
                        value,
                        file,
                        dir_path,
                        attr,
                        depth + 1,
                        max_depth,
                        1,
                        max_list_shape,
                    )


def _write_list_attributes_recursive(
    it, file, dir_path, file_name, depth, max_depth, shape, max_list_shape
):
    if shape > max_list_shape:
        return

    for idx, item in enumerate(it):
        if isinstance(item, BaseEstimator):
            new_dir_path = f"{dir_path}/{file_name}_{idx}/"
            file.write(
                f"    See {new_dir_path}{file_name}_{idx}.txt for more details\n"
            )
            _write_estimator_attributes_recursive(
                item,
                new_dir_path,
                f"{file_name}_{idx}",
                depth,
                max_depth,
                max_list_shape,
            )
        elif _is_non_string_sequence(item):
            _write_list_attributes_recursive(
                item,
                file,
                dir_path,
                f"{file_name}_{idx}",
                depth,
                max_depth,
                shape + 1,
                max_list_shape,
            )


def _is_non_string_sequence(obj):
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))
