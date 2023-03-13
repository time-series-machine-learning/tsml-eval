# -*- coding: utf-8 -*-
"""Set classifier function."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]


def set_clusterer(clusterer_name, random_state=None, n_jobs=1):
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

    Return
    ------
    clusterer: A BaseClusterer.
        The clusterer matching the input clusterer name.
    """
    c = clusterer_name.lower()

    # Distance based
    if c == "kmeans" or c == "k-means":
        from sktime.clustering.k_means import TimeSeriesKMeans

        return TimeSeriesKMeans(
            n_clusters=5,
            max_iter=50,
            random_state=random_state,
        )
    elif c == "kmedoids" or c == "k-medoids":
        from sktime.clustering.k_medoids import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            n_clusters=5,
            max_iter=50,
            random_state=random_state,
        )

    # Dummy clusterer
    elif c == "dummy" or c == "dummyclusterer" or c == "dummyclusterer-tsml":
        from tsml.dummy import DummyClusterer

        return DummyClusterer(strategy="random", random_state=random_state)
    elif c == "dummyclusterer-sktime":
        from sktime.clustering.k_means import TimeSeriesKMeans

        return TimeSeriesKMeans(
            n_clusters=2,
            metric="euclidean",
            max_iter=1,
            random_state=0,
        )
    elif c == "dummyclusterer-sklearn":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=2, max_iter=1, random_state=0)

    # invalid clusterer
    else:
        raise Exception("UNKNOWN CLUSTERER ", c, " in set_clusterer")
