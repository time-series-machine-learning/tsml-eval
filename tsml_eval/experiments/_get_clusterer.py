"""Get clusterer function."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
from aeon.clustering import (
    ElasticSOM,
    KSpectralCentroid,
    TimeSeriesCLARA,
    TimeSeriesCLARANS,
    TimeSeriesKernelKMeans,
    TimeSeriesKMeans,
    TimeSeriesKMedoids,
    TimeSeriesKShape,
)
from aeon.transformations.collection import Normalizer
from sklearn.cluster import KMeans

from tsml_eval.utils.datasets import load_experiment_data
from tsml_eval.utils.functions import str_in_nested_list

deep_learning_clusterers = [
    ["aefcnclusterer", "aefcn"],
    ["aeresnetclusterer", "aeresnet"],
    ["aeattentionbigruclusterer", "aeattentionbigru"],
    ["aebigruclusterer", "aebigru"],
    ["aedcnnclusterer", "aedcnn"],
    ["aedrnnclusterer", "aedrnn"],
]
distance_based_clusterers = [
    "kmeans-euclidean",
    "kmeans-squared",
    ["kmeans-dtw", "timeserieskmeans"],
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
    ["kmedoids-dtw", "timeserieskmedoids"],
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
    ["clarans-dtw", "timeseriesclarans"],
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
    ["clara-dtw", "timeseriesclara"],
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
    "kmeans-ssg-ba-dtw",
    "kmeans-ssg-ba-ddtw",
    "kmeans-ssg-ba-wdtw",
    "kmeans-ssg-ba-wddtw",
    "kmeans-ssg-ba-erp",
    "kmeans-ssg-ba-edr",
    "kmeans-ssg-ba-twe",
    "kmeans-ssg-ba-msm",
    "kmeans-ssg-ba-adtw",
    "kmeans-ssg-ba-shape_dtw",
    ["som-dtw", "elasticsom"],
    "som-ddtw",
    "som-wdtw",
    "som-wddtw",
    "som-lcss",
    "som-erp",
    "som-edr",
    "som-twe",
    "som-msm",
    "som-adtw",
    "som-shape_dtw",
    "som-soft_dtw",
    ["kspectralcentroid", "ksc"],
    ["timeserieskshape", "kshape"],
    "kasba",
]
feature_based_clusterers = [
    ["catch22", "catch22clusterer"],
    ["tsfresh", "tsfreshclusterer"],
    ["summary", "summaryclusterer"],
]
other_clusterers = [
    ["dummyclusterer", "dummy", "dummyclusterer-aeon"],
    "dummyclusterer-tsml",
    "dummyclusterer-sklearn",
]
vector_clusterers = [
    ["kmeans", "kmeans-sklearn"],
    "dbscan",
]


def get_clusterer_by_name(
    clusterer_name,
    random_state=None,
    n_jobs=1,
    fit_contract=0,
    checkpoint=None,
    data_vars=None,
    row_normalise=False,
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
    random_state : int, RandomState instance or None, default=None
        Random seed or RandomState object to be used in the clusterer if available.
    n_jobs: int, default=1
        The number of jobs to run in parallel for both clusterer ``fit`` and
        ``predict`` if available. `-1` means using all processors.
    fit_contract: int, default=0
        The number of data points to use in the clusterer ``fit`` if available.
    checkpoint: str, default=None
        Checkpoint to save model
    data_vars: list, default=None
        List of arguments to load the dataset using
        `tsml_eval.utils.experiments import load_experiment_data`.
    row_normalise: bool, default=False
        Whether to row normalise the data if it is loaded using data_vars.

    Return
    ------
    clusterer: A BaseClusterer.
        The clusterer matching the input clusterer name.
    """
    c = clusterer_name.lower()

    if str_in_nested_list(deep_learning_clusterers, c):
        return _set_clusterer_deep_learning(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(distance_based_clusterers, c):
        return _set_clusterer_distance_based(
            c,
            random_state,
            n_jobs,
            fit_contract,
            checkpoint,
            data_vars,
            row_normalise,
            kwargs,
        )
    elif str_in_nested_list(feature_based_clusterers, c):
        return _set_clusterer_feature_based(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
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
        raise ValueError(f"UNKNOWN CLUSTERER: {c} in get_clusterer_by_name")


def _set_clusterer_deep_learning(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "aefcnclusterer" or c == "aefcn":
        from aeon.clustering.deep_learning import AEFCNClusterer

        return AEFCNClusterer(
            estimator=TimeSeriesKMeans(distance="euclidean", averaging_method="mean"),
            random_state=random_state,
            **kwargs,
        )
    elif c == "aeresnetclusterer" or c == "aeresnet":
        from aeon.clustering.deep_learning import AEResNetClusterer

        return AEResNetClusterer(
            estimator=TimeSeriesKMeans(distance="euclidean", averaging_method="mean"),
            random_state=random_state,
            **kwargs,
        )
    elif c == "aeattentionbigruclusterer" or c == "aeattentionbigru":
        from aeon.clustering.deep_learning import AEAttentionBiGRUClusterer

        return AEAttentionBiGRUClusterer(
            estimator=TimeSeriesKMeans(distance="euclidean", averaging_method="mean"),
            random_state=random_state,
            **kwargs,
        )
    elif c == "aebigruclusterer" or c == "aebigru":
        from aeon.clustering.deep_learning import AEBiGRUClusterer

        return AEBiGRUClusterer(
            estimator=TimeSeriesKMeans(distance="euclidean", averaging_method="mean"),
            random_state=random_state,
            **kwargs,
        )
    elif c == "aedcnnclusterer" or c == "aedcnn":
        from aeon.clustering.deep_learning import AEDCNNClusterer

        return AEDCNNClusterer(
            estimator=TimeSeriesKMeans(distance="euclidean", averaging_method="mean"),
            random_state=random_state,
            **kwargs,
        )
    elif c == "aedrnnclusterer" or c == "aedrnn":
        from aeon.clustering.deep_learning import AEDRNNClusterer

        return AEDRNNClusterer(
            estimator=TimeSeriesKMeans(distance="euclidean", averaging_method="mean"),
            random_state=random_state,
            **kwargs,
        )


def _set_clusterer_distance_based(
    c,
    random_state,
    n_jobs,
    fit_contract,
    checkpoint,
    data_vars,
    row_normalise,
    kwargs,
):
    if "init_algorithm" in kwargs:
        init_algorithm = kwargs["init_algorithm"]
    else:
        init_algorithm = "random"

    if "distance" in kwargs:
        distance = kwargs["distance"]
    else:
        if "-" not in c:
            print("No distance metric specified, using default DTW")  # noqa: T201
            distance = "dtw"
        else:
            distance = c.split("-")[-1]

    if "distance_params" in kwargs:
        distance_params = kwargs["distance_params"]
    else:
        distance_params = _get_distance_default_params(
            distance, data_vars, row_normalise
        )

    if "kmeans" in c or "timeserieskmeans" in c:
        if "average_params" in kwargs:
            average_params = kwargs["average_params"]
        else:
            average_params = {"distance": distance, **distance_params.copy()}

        if "ssg" in c:
            # Sets to use subgradient BA
            average_params = {
                **average_params,
                "method": "subgradient",
            }
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=10,
                init=init_algorithm,
                distance=distance,
                distance_params=distance_params,
                random_state=random_state,
                averaging_method="ba",
                average_params=average_params,
                **kwargs,
            )
        elif "ba" in c:
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=10,
                init=init_algorithm,
                distance=distance,
                distance_params=distance_params,
                random_state=random_state,
                averaging_method="ba",
                average_params=average_params,
                **kwargs,
            )
        else:
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=10,
                init=init_algorithm,
                distance=distance,
                distance_params=distance_params,
                random_state=random_state,
                averaging_method="mean",
                **kwargs,
            )
    elif "kmedoids" in c or "timeserieskmedoids" in c:
        return TimeSeriesKMedoids(
            max_iter=50,
            n_init=10,
            init=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif "pam" in c or "timeseriespam" in c:
        return TimeSeriesKMedoids(
            max_iter=50,
            n_init=10,
            init=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif "clarans" in c or "timeseriesclarans" in c:
        return TimeSeriesCLARANS(
            n_init=10,
            init=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            **kwargs,
        )
    elif "clara" in c or "timeseriesclara" in c:
        return TimeSeriesCLARA(
            max_iter=50,
            init=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            **kwargs,
        )
    elif "som" in c or "elasticsom" in c:
        return ElasticSOM(
            distance=distance,
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            distance_params=distance_params,
            random_state=random_state,
            verbose=False,
        )
    elif "ksc" in c or "kspectralcentroid" in c:
        return KSpectralCentroid(
            # Max shift set to n_timepoints when max_shift is None
            max_shift=None,
            max_iter=50,
            init=init_algorithm,
            tol=1e-06,
            random_state=random_state,
            **kwargs,
        )
    elif "kshape" in c:
        return TimeSeriesKShape(
            init=init_algorithm,
            max_iter=50,
            n_init=10,
            tol=1e-06,
            random_state=random_state,
            **kwargs,
        )
    elif "timeserieskernelkmeans" in c:
        return TimeSeriesKernelKMeans(
            max_iter=50,
            n_init=10,
            tol=1e-06,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "kasba":
        from aeon.clustering import KASBA

        return KASBA(
            random_state=random_state,
            **kwargs,
        )


def _get_distance_default_params(
    dist_name: str, data_vars: list, row_normalise: bool
) -> dict:
    if dist_name == "dtw" or dist_name == "ddtw":
        return {"window": 0.2}
    if dist_name == "lcss":
        return {"epsilon": 1.0}
    if dist_name == "erp":
        # load dataset to get std if available
        if data_vars is not None:
            X_train, _, _, _, _ = load_experiment_data(*data_vars)

            # cant handle unequal length series
            if isinstance(X_train, np.ndarray):
                if row_normalise:
                    scaler = Normalizer()
                    X_train = scaler.fit_transform(X_train)

                return {"g": X_train.std(axis=0).sum()}
            elif not isinstance(X_train, list):
                raise ValueError("Unknown data type in _get_distance_default_params")
        return {"g": 0.05}
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


def _set_clusterer_feature_based(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "catch22" or c == "catch22clusterer":
        from aeon.clustering.feature_based import Catch22Clusterer

        return Catch22Clusterer(
            estimator=KMeans(), random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "tsfresh" or c == "tsfreshclusterer":
        from aeon.clustering.feature_based import TSFreshClusterer

        return TSFreshClusterer(
            estimator=KMeans(), random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "summary" or c == "summaryclusterer":
        from aeon.clustering.feature_based import SummaryClusterer

        return SummaryClusterer(
            estimator=KMeans(), random_state=random_state, n_jobs=n_jobs, **kwargs
        )


def _set_clusterer_other(c, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if c == "dummyclusterer" or c == "dummy" or c == "dummyclusterer-aeon":
        from aeon.clustering.dummy import DummyClusterer

        return DummyClusterer(random_state=random_state, **kwargs)
    elif c == "dummyclusterer-tsml":
        from tsml.dummy import DummyClusterer

        return DummyClusterer(strategy="random", random_state=random_state, **kwargs)
    elif c == "dummyclusterer-sklearn":
        return KMeans(
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
