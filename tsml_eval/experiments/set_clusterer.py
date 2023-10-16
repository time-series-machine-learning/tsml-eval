"""Set classifier function."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

from typing import Union

import numpy as np
from aeon.clustering import (
    TimeSeriesCLARA,
    TimeSeriesCLARANS,
    TimeSeriesKMeans,
    TimeSeriesKMedoids,
)
from sklearn.cluster import KMeans
from tsml.datasets import load_from_ts_file

from tsml_eval.estimators.transformations.scaler import TimeSeriesScaler
from tsml_eval.utils.experiments import load_clustering_experiment_data
from tsml_eval.utils.functions import str_in_nested_list


def _load_data(problem_path, dataset, resample_id, predefined_resample):
    if resample_id is not None and predefined_resample:
        resample_str = "" if resample_id is None else str(resample_id)

        X_train, y_train = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TRAIN.ts"
        )
        X_test, y_test = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TEST.ts"
        )

        resample_data = False
    else:
        X_train, y_train = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}_TRAIN.ts"
        )
        X_test, y_test = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}_TEST.ts"
        )

        resample_data = True if resample_id != 0 else False

    return X_train, y_train, X_test, y_test, resample_data


distance_based_clusterers = [
    "kmeans-euclidean",
    "kmeans-squared",
    "kmeans-dtw",
    "kmeans-ddtw",
    "kmeans-wdtw",
    "kmeans-wddtw",
    "kmeans-lcss",
    "kmeans-erp",
    "kmeans-edr",
    "kmeans-twe",
    "kmeans-msm",
    "kmeans-adtw",
    "kmeans-shape_dtw",
    "kmedoids-euclidean",
    "kmedoids-squared",
    "kmedoids-dtw",
    "kmedoids-ddtw",
    "kmedoids-wdtw",
    "kmedoids-wddtw",
    "kmedoids-lcss",
    "kmedoids-erp",
    "kmedoids-edr",
    "kmedoids-twe",
    "kmedoids-msm",
    "kmedoids-adtw",
    "kmedoids-shape_dtw",
    "clarans-euclidean",
    "clarans-squared",
    "clarans-dtw",
    "clarans-ddtw",
    "clarans-wdtw",
    "clarans-wddtw",
    "clarans-lcss",
    "clarans-erp",
    "clarans-edr",
    "clarans-twe",
    "clarans-msm",
    "clarans-adtw",
    "clarans-shape_dtw",
    "clara-euclidean",
    "clara-squared",
    "clara-dtw",
    "clara-ddtw",
    "clara-wdtw",
    "clara-wddtw",
    "clara-lcss",
    "clara-erp",
    "clara-edr",
    "clara-twe",
    "clara-msm",
    "clara-adtw",
    "clara-shape_dtw",
    "pam-euclidean",
    "pam-squared",
    "pam-dtw",
    "pam-ddtw",
    "pam-wdtw",
    "pam-wddtw",
    "pam-lcss",
    "pam-erp",
    "pam-edr",
    "pam-twe",
    "pam-msm",
    "pam-adtw",
    "pam-shape_dtw",
    "kmeans-ba-euclidean",
    "kmeans-ba-squared",
    "kmeans-ba-dtw",
    "kmeans-ba-ddtw",
    "kmeans-ba-wdtw",
    "kmeans-ba-wddtw",
    "kmeans-ba-lcss",
    "kmeans-ba-erp",
    "kmeans-ba-edr",
    "kmeans-ba-twe",
    "kmeans-ba-msm",
    "kmeans-ba-adtw",
    "kmeans-ba-shape_dtw",
    "TimeSeriesKMeans",
    "TimeSeriesKMedoids",
    "TimeSeriesCLARANS",
    "TimeSeriesCLARA",
]

other_clusterers = [
    ["DummyClusterer", "dummy", "dummyclusterer-tsml"],
    "dummyclusterer-aeon",
    "dummyclusterer-sklearn",
]
vector_clusterers = [
    ["KMeans", "kmeans-sklearn"],
    "DBSCAN",
]


def _get_distance_default_params(train_data: np.ndarray, dist_name: str) -> dict:
    if dist_name == "dtw" or dist_name == "ddtw":
        return {"window": 0.2}
    if dist_name == "lcss":
        return {"epsilon": 1.0}
    if dist_name == "erp":
        if len(train_data) > 0:
            return {"g": train_data.std(axis=0).sum()}
    if dist_name == "msm":
        return {"c": 1.0, "independent": True}
    if dist_name == "edr":
        return {"epsilon": None}
    if dist_name == "twe":
        return {"nu": 0.001, "lmbda": 1.0}
    if dist_name == "psi_dtw":
        return {"r": 0.5}
    if dist_name == "adtw":
        return {"warp_penalty": 1.0}
    if dist_name == "shape_dtw":
        return {"descriptor": "identity", "reach": 30}
    return {}


def set_clusterer(
    clusterer_name,
    data_path: str = Union[str, None],
    dataset_name: str = Union[str, None],
    resample_id: int = 0,
    predefined_resample: bool = False,
    random_state=None,
    n_jobs=1,
    fit_contract=0,
    checkpoint=None,
    row_normalise=False,
    combine_test_train_split=False,
    **kwargs,
):
    """Return a clusterer matching a given input name.

    Basic way of creating a clusterer to build using the default or alternative
    settings. This set up is to help with batch jobs for multiple problems and to
    facilitate easy reproducibility through run_clustering_experiment.

    Generally, inputting a clusterer class name will return said clusterer with
    default settings.

    Parameters
    ----------
    clusterer_name : str
        String indicating which clusterer to be returned.
    data_path : str, default=None
        Path to the data directory.
    dataset_name : str, default=None
        Name of the dataset to be loaded.
    resample_id : int, default=0
        Resample id to be used in the clusterer if available.
    predefined_resample : bool, default=False
        Whether the resample id is predefined.
    random_state : int, RandomState instance or None, default=None
        Random seed or RandomState object to be used in the clusterer if available.
    n_jobs: int, default=1
        The number of jobs to run in parallel for both clusterer ``fit`` and
        ``predict`` if available. `-1` means using all processors.
    fit_contract: int, default=0
        The number of data points to use in the clusterer ``fit`` if available.
    checkpoint: str, default=None
        Checkpoint to save model
    row_normalise: bool, default=False
        Whether to row normalise the data before fitting.
    combine_test_train_split: bool, default=False
        Whether to combine the train and test splits before fitting.

    Return
    ------
    clusterer: A BaseClusterer.
        The clusterer matching the input clusterer name.
    """
    c = clusterer_name.lower()

    if str_in_nested_list(distance_based_clusterers, c):
        X_train = np.empty(0)
        y_train = np.empty(0)
        if data_path is not None:
            X_train, y_train, _, _, _ = load_clustering_experiment_data(
                data_path,
                dataset_name,
                resample_id,
                predefined_resample,
                combine_test_train_split,
            )
            if row_normalise:
                scaler = TimeSeriesScaler()
                X_train = scaler.fit_transform(X_train)
        return _set_clusterer_distance_based(
            c, random_state, n_jobs, fit_contract, checkpoint, X_train, y_train, kwargs
        )
    elif str_in_nested_list(other_clusterers, c):
        return _set_clusterer_other(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(vector_clusterers, c):
        return _set_clusterer_vector(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    else:
        raise ValueError(f"UNKNOWN CLUSTERER: {c} in set_clusterer")


def _set_clusterer_distance_based(
    c, random_state, n_jobs, fit_contract, checkpoint, X_train, y_train, kwargs
):
    if "init_algorithm" in kwargs:
        init_algorithm = kwargs["init_algorithm"]
    else:
        init_algorithm = "random"

    if "distance" in kwargs:
        distance = kwargs["distance"]
    else:
        distance = c.split("-")[-1]

    if "distance_params" in kwargs:
        distance_params = kwargs["distance_params"]
    else:
        distance_params = _get_distance_default_params(X_train, distance)

    if "n_clusters" in kwargs:
        n_clusters = kwargs["n_clusters"]
    else:
        n_clusters = len(np.unique(y_train))

    if "kmeans" in c or "timeserieskmeans" in c:
        if "average_params" in kwargs:
            average_params = kwargs["average_params"]
        else:
            average_params = {"distance": distance, **distance_params.copy()}
        if "ba" in c:
            return TimeSeriesKMeans(
                n_clusters=n_clusters,
                init_algorithm=init_algorithm,
                distance=distance,
                distance_params=distance_params,
                random_state=random_state,
                averaging_method="ba",
                average_params=average_params,
                **kwargs,
            )
        else:
            return TimeSeriesKMeans(
                n_clusters=n_clusters,
                init_algorithm=init_algorithm,
                distance=distance,
                distance_params=distance_params,
                random_state=random_state,
                **kwargs,
            )
    elif "kmedoids" in c or "timeserieskmedoids" in c:
        return TimeSeriesKMedoids(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif "pam" in c or "timeseriespam" in c:
        return TimeSeriesKMedoids(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif "clarans" in c or "timeseriesclarans" in c:
        return TimeSeriesCLARANS(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            **kwargs,
        )
    elif "clara" in c or "timeseriesclara" in c:
        return TimeSeriesCLARA(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            **kwargs,
        )
    return None


def _set_clusterer_other(c, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if c == "dummyclusterer" or c == "dummy" or c == "dummyclusterer-tsml":
        from tsml.dummy import DummyClusterer

        return DummyClusterer(
            strategy="random", n_clusters=1, random_state=random_state, **kwargs
        )
    elif c == "dummyclusterer-aeon":
        return TimeSeriesKMeans(
            n_clusters=1,
            n_init=1,
            init_algorithm="random",
            distance="euclidean",
            max_iter=1,
            random_state=random_state,
            **kwargs,
        )
    elif c == "dummyclusterer-sklearn":
        return KMeans(
            n_clusters=1,
            n_init=1,
            init="random",
            max_iter=1,
            random_state=random_state,
            **kwargs,
        )


def _set_clusterer_vector(c, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if c == "kmeans" or c == "kmeans-sklearn":
        from sklearn.cluster import KMeans

        return KMeans(random_state=random_state, **kwargs)
    elif c == "dbscan":
        from sklearn.cluster import DBSCAN

        return DBSCAN(n_jobs=n_jobs, **kwargs)
