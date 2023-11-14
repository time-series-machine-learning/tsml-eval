"""Experiment runner for distance clustering publication."""

__author__ = ["MatthewMiddlehurst"]

import os
import sys

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

from tsml_eval.experiments import load_and_run_clustering_experiment
from tsml_eval.publications.y2023.distance_based_clustering.set_distance_clusterer import (  # noqa: E501
    _set_distance_clusterer,
)
from tsml_eval.publications.y2023.distance_based_clustering.tests import (
    _DISTANCE_TEST_RESULTS_PATH,
)
from tsml_eval.testing.test_utils import _TEST_DATA_PATH
from tsml_eval.utils.arguments import parse_args
from tsml_eval.utils.experiments import _results_present

classifiers = [
    "KMeans-dtw",
    "KMeans-ddtw",
    "KMeans-ed",
    "KMeans-edr",
    "KMeans-erp",
    "KMeans-lcss",
    "KMeans-msm",
    "KMeans-twe",
    "KMeans-wdtw",
    "KMeans-wddtw",
    "KMedoids-dtw",
    "KMedoids-ddtw",
    "KMedoids-ed",
    "KMedoids-edr",
    "KMedoids-erp",
    "KMedoids-lcss",
    "KMedoids-msm",
    "KMedoids-twe",
    "KMedoids-wdtw",
    "KMedoids-wddtw",
]


def _run_experiment(args):
    if args is None or args.__len__() < 1:
        data_path = _TEST_DATA_PATH
        results_path = _DISTANCE_TEST_RESULTS_PATH
        clusterer_name = "KMeans-dtw"
        dataset_name = "MinimalChinatown"
        resample_id = 0
        normalise = False
        kwargs = {}
        overwrite = False
    else:
        print("Input args = ", args)
        args = parse_args(args)
        data_path = args.data_path
        results_path = args.results_path
        clusterer_name = args.estimator_name
        dataset_name = args.dataset_name
        resample_id = args.resample_id
        normalise = args.row_normalise
        kwargs = args.kwargs
        overwrite = args.overwrite

    distance = clusterer_name.split("-")[-1]

    # further default parameterisation for clusterers and distances.
    # feel free to change
    kwargs["init_algorithm"] = "random"
    kwargs["max_iter"] = 30
    kwargs["n_init"] = 10

    distance_params = {
        "window": 0.2 if distance == "dtw" or distance == "wdtw" else 1.0,
        "epsilon": 0.05,
        "g": 0.05,
        "c": 1.0,
        "nu": 0.05,
        "lmbda": 1.0,
        "strategy": "independent",
    }
    kwargs["distance_params"] = distance_params

    cnl = clusterer_name.lower()
    if cnl.find("kmeans") or cnl.find("k-means"):
        kwargs["averaging_method"] = "mean"
        average_params = {
            **distance_params,
            "averaging_distance_metric": distance,
            "medoids_distance_metric": distance,
        }
        kwargs["average_params"] = average_params

    # Skip if not overwrite and results already present
    # this is also checked in load_and_run, but doing a quick check here so can
    # print a message and make sure data is not loaded
    if not overwrite and _results_present(
        results_path,
        clusterer_name,
        dataset_name,
        resample_id=resample_id,
        split="BOTH",
    ):
        print("Ignoring, results already present")
    else:
        load_and_run_clustering_experiment(
            data_path,
            results_path,
            dataset_name,
            _set_distance_clusterer(
                clusterer_name,
                random_state=resample_id + 1,
                **kwargs,
            ),
            row_normalise=normalise,
            n_clusters=-1,
            clusterer_name=clusterer_name,
            resample_id=resample_id,
            build_test_file=True,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    print("Running run_distance_experiments.py main")
    _run_experiment(sys.argv[1:])
