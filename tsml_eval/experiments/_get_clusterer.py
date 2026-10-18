"""Get clusterer function."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
from aeon.clustering import (
    ElasticSOM,
    KShape,
    KSpectralCentroid,
    TimeSeriesCLARA,
    TimeSeriesCLARANS,
    TimeSeriesKernelKMeans,
    TimeSeriesKMeans,
    TimeSeriesKMedoids,
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
    "timeserieskernelkmeans",
    "kasba",
    "agglomerative-euclidean",
    "agglomerative-squared",
    ["agglomerative-dtw", "timeseriesagglomerative"],
    "agglomerative-ddtw",
    "agglomerative-wdtw",
    "agglomerative-wddtw",
    "agglomerative-lcss",
    "agglomerative-erp",
    "agglomerative-edr",
    "agglomerative-twe",
    "agglomerative-msm",
    "agglomerative-adtw",
    "agglomerative-shape_dtw",
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
    **kwargs
        Additional keyword arguments to be passed to the clusterer.

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
    if c == "kmeans-euclidean":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="euclidean",
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-squared":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="squared",
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-dtw" or c == "timeserieskmeans":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-ddtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-wdtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="wdtw",
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-wddtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="wddtw",
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-lcss":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="lcss",
            distance_params={"epsilon": 1.0},
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-erp":
        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-edr":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="edr",
            distance_params={"epsilon": None},
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-twe":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="twe",
            distance_params={"nu": 0.001, "lmbda": 1.0},
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-msm":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="msm",
            distance_params={"c": 1.0, "independent": True},
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-adtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="adtw",
            distance_params={"warp_penalty": 1.0},
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-shape_dtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="shape_dtw",
            distance_params={"descriptor": "identity", "reach": 30},
            random_state=random_state,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmedoids-euclidean":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="euclidean",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-squared":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="squared",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-dtw" or c == "timeserieskmedoids":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-ddtw":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-wdtw":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="wdtw",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-wddtw":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="wddtw",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-lcss":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="lcss",
            distance_params={"epsilon": 1.0},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-erp":
        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-edr":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="edr",
            distance_params={"epsilon": None},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-twe":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="twe",
            distance_params={"nu": 0.001, "lmbda": 1.0},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-msm":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="msm",
            distance_params={"c": 1.0, "independent": True},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-adtw":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="adtw",
            distance_params={"warp_penalty": 1.0},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-shape_dtw":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="shape_dtw",
            distance_params={"descriptor": "identity", "reach": 30},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "clarans-euclidean":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="euclidean",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-squared":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="squared",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-dtw" or c == "timeseriesclarans":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-ddtw":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-wdtw":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="wdtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-wddtw":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="wddtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-lcss":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="lcss",
            distance_params={"epsilon": 1.0},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-erp":
        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-edr":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="edr",
            distance_params={"epsilon": None},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-twe":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="twe",
            distance_params={"nu": 0.001, "lmbda": 1.0},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-msm":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="msm",
            distance_params={"c": 1.0, "independent": True},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-adtw":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="adtw",
            distance_params={"warp_penalty": 1.0},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-shape_dtw":
        return TimeSeriesCLARANS(
            n_init=10,
            init="random",
            distance="shape_dtw",
            distance_params={"descriptor": "identity", "reach": 30},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-euclidean":
        return TimeSeriesCLARA(
            init="random",
            distance="euclidean",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-squared":
        return TimeSeriesCLARA(
            init="random",
            distance="squared",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-dtw" or c == "timeseriesclara":
        return TimeSeriesCLARA(
            init="random",
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-ddtw":
        return TimeSeriesCLARA(
            init="random",
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-wdtw":
        return TimeSeriesCLARA(
            init="random",
            distance="wdtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-wddtw":
        return TimeSeriesCLARA(
            init="random",
            distance="wddtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-lcss":
        return TimeSeriesCLARA(
            init="random",
            distance="lcss",
            distance_params={"epsilon": 1.0},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-erp":
        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesCLARA(
            init="random",
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-edr":
        return TimeSeriesCLARA(
            init="random",
            distance="edr",
            distance_params={"epsilon": None},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-twe":
        return TimeSeriesCLARA(
            init="random",
            distance="twe",
            distance_params={"nu": 0.001, "lmbda": 1.0},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-msm":
        return TimeSeriesCLARA(
            init="random",
            distance="msm",
            distance_params={"c": 1.0, "independent": True},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-adtw":
        return TimeSeriesCLARA(
            init="random",
            distance="adtw",
            distance_params={"warp_penalty": 1.0},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-shape_dtw":
        return TimeSeriesCLARA(
            init="random",
            distance="shape_dtw",
            distance_params={"descriptor": "identity", "reach": 30},
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-euclidean":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="euclidean",
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-squared":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="squared",
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-dtw":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-ddtw":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-wdtw":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="wdtw",
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-wddtw":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="wddtw",
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-lcss":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="lcss",
            distance_params={"epsilon": 1.0},
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-erp":
        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-edr":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="edr",
            distance_params={"epsilon": None},
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-twe":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="twe",
            distance_params={"nu": 0.001, "lmbda": 1.0},
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-msm":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="msm",
            distance_params={"c": 1.0, "independent": True},
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-adtw":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="adtw",
            distance_params={"warp_penalty": 1.0},
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "pam-shape_dtw":
        return TimeSeriesKMedoids(
            n_init=10,
            init="random",
            distance="shape_dtw",
            distance_params={"descriptor": "identity", "reach": 30},
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif c == "kmeans-ba-dtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "dtw", "window": 0.2},
            **kwargs,
        )
    elif c == "kmeans-ba-ddtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "ddtw", "window": 0.2},
            **kwargs,
        )
    elif c == "kmeans-ba-wdtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="wdtw",
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "wdtw"},
            **kwargs,
        )
    elif c == "kmeans-ba-wddtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="wddtw",
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "wddtw"},
            **kwargs,
        )
    elif c == "kmeans-ba-lcss":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="lcss",
            distance_params={"epsilon": 1.0},
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "lcss", "epsilon": 1.0},
            **kwargs,
        )
    elif c == "kmeans-ba-erp":
        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "erp", "g": g},
            **kwargs,
        )
    elif c == "kmeans-ba-edr":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="edr",
            distance_params={"epsilon": None},
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "edr", "epsilon": None},
            **kwargs,
        )
    elif c == "kmeans-ba-twe":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="twe",
            distance_params={"nu": 0.001, "lmbda": 1.0},
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "twe", "nu": 0.001, "lmbda": 1.0},
            **kwargs,
        )
    elif c == "kmeans-ba-msm":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="msm",
            distance_params={"c": 1.0, "independent": True},
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "msm", "c": 1.0, "independent": True},
            **kwargs,
        )
    elif c == "kmeans-ba-adtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="adtw",
            distance_params={"warp_penalty": 1.0},
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "adtw", "warp_penalty": 1.0},
            **kwargs,
        )
    elif c == "kmeans-ba-shape_dtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="shape_dtw",
            distance_params={"descriptor": "identity", "reach": 30},
            random_state=random_state,
            averaging_method="ba",
            average_params={
                "distance": "shape_dtw",
                "descriptor": "identity",
                "reach": 30,
            },
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-dtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "dtw", "window": 0.2, "method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-ddtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "ddtw", "window": 0.2, "method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-wdtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="wdtw",
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "wdtw", "method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-wddtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="wddtw",
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "wddtw", "method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-erp":
        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            averaging_method="ba",
            average_params={"distance": "erp", "g": g, "method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-edr":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="edr",
            distance_params={"epsilon": None},
            random_state=random_state,
            averaging_method="ba",
            average_params={
                "distance": "edr",
                "epsilon": None,
                "method": "subgradient",
            },
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-twe":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="twe",
            distance_params={"nu": 0.001, "lmbda": 1.0},
            random_state=random_state,
            averaging_method="ba",
            average_params={
                "distance": "twe",
                "nu": 0.001,
                "lmbda": 1.0,
                "method": "subgradient",
            },
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-msm":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="msm",
            distance_params={"c": 1.0, "independent": True},
            random_state=random_state,
            averaging_method="ba",
            average_params={
                "distance": "msm",
                "c": 1.0,
                "independent": True,
                "method": "subgradient",
            },
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-adtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="adtw",
            distance_params={"warp_penalty": 1.0},
            random_state=random_state,
            averaging_method="ba",
            average_params={
                "distance": "adtw",
                "warp_penalty": 1.0,
                "method": "subgradient",
            },
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-shape_dtw":
        return TimeSeriesKMeans(
            n_init=10,
            init="random",
            distance="shape_dtw",
            distance_params={"descriptor": "identity", "reach": 30},
            random_state=random_state,
            averaging_method="ba",
            average_params={
                "distance": "shape_dtw",
                "descriptor": "identity",
                "reach": 30,
                "method": "subgradient",
            },
            **kwargs,
        )
    elif c == "som-dtw" or c == "elasticsom":
        return ElasticSOM(
            distance="dtw",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            distance_params={"window": 0.2},
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "som-ddtw":
        return ElasticSOM(
            distance="ddtw",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            distance_params={"window": 0.2},
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "som-wdtw":
        return ElasticSOM(
            distance="wdtw",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "som-wddtw":
        return ElasticSOM(
            distance="wddtw",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "som-lcss":
        return ElasticSOM(
            distance="lcss",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            distance_params={"epsilon": 1.0},
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "som-erp":
        g = _get_erp_g(data_vars, row_normalise)
        return ElasticSOM(
            distance="erp",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            distance_params={"g": g},
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "som-edr":
        return ElasticSOM(
            distance="edr",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            distance_params={"epsilon": None},
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "som-twe":
        return ElasticSOM(
            distance="twe",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            distance_params={"nu": 0.001, "lmbda": 1.0},
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "som-msm":
        return ElasticSOM(
            distance="msm",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            distance_params={"c": 1.0, "independent": True},
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "som-adtw":
        return ElasticSOM(
            distance="adtw",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            distance_params={"warp_penalty": 1.0},
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "som-shape_dtw":
        return ElasticSOM(
            distance="shape_dtw",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            distance_params={"descriptor": "identity", "reach": 30},
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "som-soft_dtw":
        return ElasticSOM(
            distance="soft_dtw",
            init="random",
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            sigma_decay_function="asymptotic_decay",
            num_iterations=500,
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
    elif c == "kspectralcentroid" or c == "ksc":
        return KSpectralCentroid(
            # Max shift set to n_timepoints when max_shift is None
            max_shift=None,
            init="random",
            tol=1e-06,
            random_state=random_state,
            **kwargs,
        )
    elif c == "timeserieskshape" or c == "kshape":
        return KShape(
            init="random",
            n_init=10,
            tol=1e-06,
            random_state=random_state,
            **kwargs,
        )
    elif c == "timeserieskernelkmeans":
        return TimeSeriesKernelKMeans(
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
    elif c == "agglomerative-euclidean":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="euclidean",
            **kwargs,
        )
    elif c == "agglomerative-squared":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="squared",
            **kwargs,
        )
    elif c == "agglomerative-dtw" or c == "timeseriesagglomerative":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="dtw",
            distance_params={"window": 0.2},
            **kwargs,
        )
    elif c == "agglomerative-ddtw":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="ddtw",
            distance_params={"window": 0.2},
            **kwargs,
        )
    elif c == "agglomerative-wdtw":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="wdtw",
            **kwargs,
        )
    elif c == "agglomerative-wddtw":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="wddtw",
            **kwargs,
        )
    elif c == "agglomerative-lcss":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="lcss",
            distance_params={"epsilon": 1.0},
            **kwargs,
        )
    elif c == "agglomerative-erp":
        from aeon.clustering import TimeSeriesAgglomerative

        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesAgglomerative(
            distance="erp",
            distance_params={"g": g},
            **kwargs,
        )
    elif c == "agglomerative-edr":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="edr",
            distance_params={"epsilon": None},
            **kwargs,
        )
    elif c == "agglomerative-twe":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="twe",
            distance_params={"nu": 0.001, "lmbda": 1.0},
            **kwargs,
        )
    elif c == "agglomerative-msm":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="msm",
            distance_params={"c": 1.0, "independent": True},
            **kwargs,
        )
    elif c == "agglomerative-adtw":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="adtw",
            distance_params={"warp_penalty": 1.0},
            **kwargs,
        )
    elif c == "agglomerative-shape_dtw":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="shape_dtw",
            distance_params={"descriptor": "identity", "reach": 30},
            **kwargs,
        )
    else:
        raise ValueError(f"UNKNOWN CLUSTERER: {c} in get_clusterer_by_name")


def _get_erp_g(data_vars: list, row_normalise: bool) -> float:
    # load dataset to get std if available
    if data_vars is not None:
        X_train, _, _, _, _ = load_experiment_data(*data_vars)

        # can't handle unequal length series
        if isinstance(X_train, np.ndarray):
            if row_normalise:
                scaler = Normalizer()
                X_train = scaler.fit_transform(X_train)

            return X_train.std(axis=0).sum()
        elif not isinstance(X_train, list):
            raise ValueError("Unknown data type in _get_erp_g")
    return 0.05


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
