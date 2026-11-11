"""Get clusterer function."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
from aeon.transformations.collection import Normalizer

from tsml_eval.utils.datasets import load_experiment_data
from tsml_eval.utils.functions import str_in_nested_list
from tsml_eval.utils.resampling import stratified_resample_data

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
    "kmeans-ba-edr",
    "kmeans-ba-twe",
    "kmeans-ba-msm",
    "kmeans-ba-adtw",
    "kmeans-ba-shape_dtw",
    "kmeans-ssg-ba-dtw",
    "kmeans-ssg-ba-ddtw",
    "kmeans-ssg-ba-wdtw",
    "kmeans-ssg-ba-wddtw",
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
        `tsml_eval.utils.datasets.load_experiment_data`, used to find the ERP g
        parameter from the train data. An optional fifth item is True if the train
        and test data are combined.
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
        from aeon.clustering import TimeSeriesKMeans
        from aeon.clustering.deep_learning import AEFCNClusterer

        return AEFCNClusterer(
            estimator=TimeSeriesKMeans(distance="euclidean", averaging_method="mean"),
            random_state=random_state,
            **kwargs,
        )
    elif c == "aeresnetclusterer" or c == "aeresnet":
        from aeon.clustering import TimeSeriesKMeans
        from aeon.clustering.deep_learning import AEResNetClusterer

        return AEResNetClusterer(
            estimator=TimeSeriesKMeans(distance="euclidean", averaging_method="mean"),
            random_state=random_state,
            **kwargs,
        )
    elif c == "aeattentionbigruclusterer" or c == "aeattentionbigru":
        from aeon.clustering import TimeSeriesKMeans
        from aeon.clustering.deep_learning import AEAttentionBiGRUClusterer

        return AEAttentionBiGRUClusterer(
            estimator=TimeSeriesKMeans(distance="euclidean", averaging_method="mean"),
            random_state=random_state,
            **kwargs,
        )
    elif c == "aebigruclusterer" or c == "aebigru":
        from aeon.clustering import TimeSeriesKMeans
        from aeon.clustering.deep_learning import AEBiGRUClusterer

        return AEBiGRUClusterer(
            estimator=TimeSeriesKMeans(distance="euclidean", averaging_method="mean"),
            random_state=random_state,
            **kwargs,
        )
    elif c == "aedcnnclusterer" or c == "aedcnn":
        from aeon.clustering import TimeSeriesKMeans
        from aeon.clustering.deep_learning import AEDCNNClusterer

        return AEDCNNClusterer(
            estimator=TimeSeriesKMeans(distance="euclidean", averaging_method="mean"),
            random_state=random_state,
            **kwargs,
        )
    elif c == "aedrnnclusterer" or c == "aedrnn":
        from aeon.clustering import TimeSeriesKMeans
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
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="euclidean",
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-squared":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="squared",
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-dtw" or c == "timeserieskmeans":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-ddtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-wdtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="wdtw",
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-wddtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="wddtw",
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-lcss":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="lcss",
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-erp":
        from aeon.clustering import TimeSeriesKMeans

        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesKMeans(
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-edr":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="edr",
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-twe":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="twe",
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-msm":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="msm",
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-adtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="adtw",
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmeans-shape_dtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="shape_dtw",
            distance_params={"reach": 30},
            random_state=random_state,
            n_jobs=n_jobs,
            averaging_method="mean",
            **kwargs,
        )
    elif c == "kmedoids-euclidean":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="euclidean",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-squared":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="squared",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-dtw" or c == "timeserieskmedoids":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-ddtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-wdtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="wdtw",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-wddtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="wddtw",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-lcss":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="lcss",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-erp":
        from aeon.clustering import TimeSeriesKMedoids

        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesKMedoids(
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-edr":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="edr",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-twe":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="twe",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-msm":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="msm",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-adtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="adtw",
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "kmedoids-shape_dtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="shape_dtw",
            distance_params={"reach": 30},
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif c == "clarans-euclidean":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="euclidean",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-squared":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="squared",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-dtw" or c == "timeseriesclarans":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-ddtw":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-wdtw":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="wdtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-wddtw":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="wddtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-lcss":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="lcss",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-erp":
        from aeon.clustering import TimeSeriesCLARANS

        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesCLARANS(
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-edr":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="edr",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-twe":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="twe",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-msm":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="msm",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-adtw":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="adtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clarans-shape_dtw":
        from aeon.clustering import TimeSeriesCLARANS

        return TimeSeriesCLARANS(
            distance="shape_dtw",
            distance_params={"reach": 30},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-euclidean":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="euclidean",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-squared":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="squared",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-dtw" or c == "timeseriesclara":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-ddtw":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-wdtw":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="wdtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-wddtw":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="wddtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-lcss":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="lcss",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-erp":
        from aeon.clustering import TimeSeriesCLARA

        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesCLARA(
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-edr":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="edr",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-twe":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="twe",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-msm":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="msm",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-adtw":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="adtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "clara-shape_dtw":
        from aeon.clustering import TimeSeriesCLARA

        return TimeSeriesCLARA(
            distance="shape_dtw",
            distance_params={"reach": 30},
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-euclidean":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="euclidean",
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-squared":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="squared",
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-dtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-ddtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-wdtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="wdtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-wddtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="wddtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-lcss":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="lcss",
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-erp":
        from aeon.clustering import TimeSeriesKMedoids

        g = _get_erp_g(data_vars, row_normalise)
        return TimeSeriesKMedoids(
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-edr":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="edr",
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-twe":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="twe",
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-msm":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="msm",
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-adtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="adtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "pam-shape_dtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="shape_dtw",
            distance_params={"reach": 30},
            random_state=random_state,
            **kwargs,
        )
    elif c == "kmeans-ba-dtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"window": 0.2},
            **kwargs,
        )
    elif c == "kmeans-ba-ddtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"window": 0.2},
            **kwargs,
        )
    elif c == "kmeans-ba-wdtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="wdtw",
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "kmeans-ba-wddtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="wddtw",
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "kmeans-ba-edr":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="edr",
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "kmeans-ba-twe":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="twe",
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "kmeans-ba-msm":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="msm",
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "kmeans-ba-adtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="adtw",
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "kmeans-ba-shape_dtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="shape_dtw",
            distance_params={"reach": 30},
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"reach": 30},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-dtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"window": 0.2, "method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-ddtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"window": 0.2, "method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-wdtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="wdtw",
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-wddtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="wddtw",
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-edr":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="edr",
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-twe":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="twe",
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-msm":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="msm",
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-adtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="adtw",
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"method": "subgradient"},
            **kwargs,
        )
    elif c == "kmeans-ssg-ba-shape_dtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="shape_dtw",
            distance_params={"reach": 30},
            random_state=random_state,
            n_jobs=n_jobs,
            average_params={"reach": 30, "method": "subgradient"},
            **kwargs,
        )
    elif c == "som-dtw" or c == "elasticsom":
        from aeon.clustering import ElasticSOM

        return ElasticSOM(
            distance="dtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "som-ddtw":
        from aeon.clustering import ElasticSOM

        return ElasticSOM(
            distance="ddtw",
            distance_params={"window": 0.2},
            random_state=random_state,
            **kwargs,
        )
    elif c == "som-wdtw":
        from aeon.clustering import ElasticSOM

        return ElasticSOM(
            distance="wdtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "som-wddtw":
        from aeon.clustering import ElasticSOM

        return ElasticSOM(
            distance="wddtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "som-lcss":
        from aeon.clustering import ElasticSOM

        return ElasticSOM(
            distance="lcss",
            random_state=random_state,
            **kwargs,
        )
    elif c == "som-erp":
        from aeon.clustering import ElasticSOM

        g = _get_erp_g(data_vars, row_normalise)
        return ElasticSOM(
            distance="erp",
            distance_params={"g": g},
            random_state=random_state,
            **kwargs,
        )
    elif c == "som-edr":
        from aeon.clustering import ElasticSOM

        return ElasticSOM(
            distance="edr",
            random_state=random_state,
            **kwargs,
        )
    elif c == "som-twe":
        from aeon.clustering import ElasticSOM

        return ElasticSOM(
            distance="twe",
            random_state=random_state,
            **kwargs,
        )
    elif c == "som-msm":
        from aeon.clustering import ElasticSOM

        return ElasticSOM(
            distance="msm",
            random_state=random_state,
            **kwargs,
        )
    elif c == "som-adtw":
        from aeon.clustering import ElasticSOM

        return ElasticSOM(
            distance="adtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "som-shape_dtw":
        from aeon.clustering import ElasticSOM

        return ElasticSOM(
            distance="shape_dtw",
            distance_params={"reach": 30},
            random_state=random_state,
            **kwargs,
        )
    elif c == "som-soft_dtw":
        from aeon.clustering import ElasticSOM

        return ElasticSOM(
            distance="soft_dtw",
            random_state=random_state,
            **kwargs,
        )
    elif c == "kspectralcentroid" or c == "ksc":
        from aeon.clustering import KSpectralCentroid

        return KSpectralCentroid(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "timeserieskshape" or c == "kshape":
        from aeon.clustering import KShape

        return KShape(
            n_init=10,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "timeserieskernelkmeans":
        from aeon.clustering import TimeSeriesKernelKMeans

        return TimeSeriesKernelKMeans(
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
            **kwargs,
        )
    elif c == "agglomerative-twe":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="twe",
            **kwargs,
        )
    elif c == "agglomerative-msm":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="msm",
            **kwargs,
        )
    elif c == "agglomerative-adtw":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="adtw",
            **kwargs,
        )
    elif c == "agglomerative-shape_dtw":
        from aeon.clustering import TimeSeriesAgglomerative

        return TimeSeriesAgglomerative(
            distance="shape_dtw",
            distance_params={"reach": 30},
            **kwargs,
        )
    else:
        raise ValueError(f"UNKNOWN CLUSTERER: {c} in get_clusterer_by_name")


def _get_erp_g(data_vars: list, row_normalise: bool) -> float:
    # load the experiment train data to get std if available
    if data_vars is not None:
        X_train, y_train, X_test, y_test, resample = load_experiment_data(
            *data_vars[:4]
        )
        if resample:
            X_train, y_train, X_test, y_test = stratified_resample_data(
                X_train, y_train, X_test, y_test, random_state=data_vars[2]
            )
        if len(data_vars) > 4 and data_vars[4]:
            X_train = (
                np.concatenate([X_train, X_test], axis=0)
                if isinstance(X_train, np.ndarray)
                else X_train + X_test
            )

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
        from sklearn.cluster import KMeans

        return Catch22Clusterer(
            estimator=KMeans(), random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "tsfresh" or c == "tsfreshclusterer":
        from aeon.clustering.feature_based import TSFreshClusterer
        from sklearn.cluster import KMeans

        return TSFreshClusterer(
            estimator=KMeans(), random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "summary" or c == "summaryclusterer":
        from aeon.clustering.feature_based import SummaryClusterer
        from sklearn.cluster import KMeans

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
        from sklearn.cluster import KMeans

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
