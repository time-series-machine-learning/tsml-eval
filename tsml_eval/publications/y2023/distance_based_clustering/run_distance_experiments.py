"""Experiment runner for distance clustering publication."""

__maintainer__ = ["MatthewMiddlehurst"]

import os
import sys

from aeon.transformations.collection import Normalizer

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

from tsml.base import _clone_estimator

from tsml_eval.experiments import load_and_run_clustering_experiment
from tsml_eval.publications.y2023.distance_based_clustering.set_distance_clusterer import (  # noqa: E501
    _set_distance_clusterer,
)
from tsml_eval.publications.y2023.distance_based_clustering.tests import (
    _DISTANCE_TEST_RESULTS_PATH,
)
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH
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
        clusterer = "KMeans-dtw"
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
        clusterer = args.estimator_name
        dataset_name = args.dataset_name
        resample_id = args.resample_id
        normalise = args.row_normalise
        kwargs = args.kwargs
        overwrite = args.overwrite

    distance = clusterer.split("-")[-1]

    # further default parameterisation for clusterers and distances.
    # feel free to change
    kwargs["init"] = "random"
    kwargs["max_iter"] = 30
    kwargs["n_init"] = 10

    if distance == "dtw" or distance == "ddtw":
        distance_params = {"window": 0.2}
    elif distance == "wdtw" or distance == "wddtw":
        distance_params = {"g": 0.05}
    elif distance == "lcss" or distance == "edr":
        distance_params = {"epsilon": 0.05}
    elif distance == "erp":
        distance_params = {"g": 0.05}
    elif distance == "msm":
        distance_params = {"c": 1.0, "independent": True}
    elif distance == "twe":
        distance_params = {
            "nu": 0.05,
            "lmbda": 1.0,
        }
    else:
        distance_params = {}
        print("Unknown distance metric, using defaults")  # noqa: T001

    kwargs["distance_params"] = distance_params

    cnl = clusterer.lower()
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
        clusterer,
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
            (
                _set_distance_clusterer(
                    clusterer,
                    random_state=resample_id + 1,
                    **kwargs,
                )
                if isinstance(clusterer, str)
                else _clone_estimator(clusterer, resample_id)
            ),
            n_clusters=-1,
            clusterer_name=clusterer,
            resample_id=resample_id,
            data_transforms=Normalizer() if normalise else None,
            build_test_file=True,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.

    1. Edit the arguments from line 49-56 to suit your experiment, The most important
       are:
         data_path: the path to the data
         results_path: the path to the results
         clusterer: the name of the clusterer to use (check set_distance_clusterer.py),
         or an estimator object
         resample_id: the data resample id and random seed to use
    2. Run the script, if the experiment runs successfully a set of folders and a
       results csv file will be created in the results path.

    For evaluation of the written results, you can use the evaluation package, see
    our examples for usage:
    https://github.com/time-series-machine-learning/tsml-eval/blob/main/examples/

    For using your own clusterer, any clusterer following the sklearn, aeon,
    or tsml interface should be compatible with this file.
    """
    print("Running run_distance_experiments.py main")
    _run_experiment(sys.argv[1:])
