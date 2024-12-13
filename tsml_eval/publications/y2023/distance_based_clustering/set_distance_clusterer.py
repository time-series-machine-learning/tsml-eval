"""Clusterers used in the publication."""

__maintainer__ = ["MatthewMiddlehurst"]

from tsml_eval.utils.functions import str_in_nested_list

distance_based_clusterers = [
    ["TimeSeriesKMeans", "kmeans-dtw", "k-means-dtw"],
    ["kmeans-ddtw", "k-means-ddtw"],
    ["kmeans-ed", "k-means-ed"],
    ["kmeans-edr", "k-means-edr"],
    ["kmeans-erp", "k-means-erp"],
    ["kmeans-lcss", "k-means-lcss"],
    ["kmeans-msm", "k-means-msm"],
    ["kmeans-twe", "k-means-twe"],
    ["kmeans-wdtw", "k-means-wdtw"],
    ["kmeans-wddtw", "k-means-wddtw"],
    ["TimeSeriesKMedoids", "kmedoids-dtw", "k-medoids-dtw"],
    ["kmedoids-ddtw", "k-medoids-ddtw"],
    ["kmedoids-ed", "k-medoids-ed"],
    ["kmedoids-edr", "k-medoids-edr"],
    ["kmedoids-erp", "k-medoids-erp"],
    ["kmedoids-lcss", "k-medoids-lcss"],
    ["kmedoids-msm", "k-medoids-msm"],
    ["kmedoids-twe", "k-medoids-twe"],
    ["kmedoids-wdtw", "k-medoids-wdtw"],
    ["kmedoids-wddtw", "k-medoids-wddtw"],
]


def _set_distance_clusterer(
    clusterer_name,
    random_state=None,
    **kwargs,
):
    c = clusterer_name.lower()

    if not str_in_nested_list(distance_based_clusterers, c):
        raise ValueError(f"UNKNOWN CLUSTERER: {c} in set_distance_clusterer")

    if c == "timeserieskmeans" or c == "kmeans-dtw" or c == "k-means-dtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(distance="dtw", random_state=random_state, **kwargs)
    elif c == "kmeans-ddtw" or c == "k-means-ddtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(distance="ddtw", random_state=random_state, **kwargs)
    elif c == "kmeans-ed" or c == "k-means-ed":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(
            distance="euclidean", random_state=random_state, **kwargs
        )
    elif c == "kmeans-edr" or c == "k-means-edr":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(distance="edr", random_state=random_state, **kwargs)
    elif c == "kmeans-erp" or c == "k-means-erp":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(distance="erp", random_state=random_state, **kwargs)
    elif c == "kmeans-lcss" or c == "k-means-lcss":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(distance="lcss", random_state=random_state, **kwargs)
    elif c == "kmeans-msm" or c == "k-means-msm":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(distance="msm", random_state=random_state, **kwargs)
    elif c == "kmeans-twe" or c == "k-means-twe":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(distance="twe", random_state=random_state, **kwargs)
    elif c == "kmeans-wdtw" or c == "k-means-wdtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(distance="wdtw", random_state=random_state, **kwargs)
    elif c == "kmeans-wddtw" or c == "k-means-wddtw":
        from aeon.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(distance="wddtw", random_state=random_state, **kwargs)
    elif c == "timeserieskmedoids" or c == "kmedoids-dtw" or c == "k-medoids-dtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(distance="dtw", random_state=random_state, **kwargs)
    elif c == "kmedoids-ddtw" or c == "k-medoids-ddtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(distance="ddtw", random_state=random_state, **kwargs)
    elif c == "kmedoids-ed" or c == "k-medoids-ed":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(
            distance="euclidean", random_state=random_state, **kwargs
        )
    elif c == "kmedoids-edr" or c == "k-medoids-edr":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(distance="edr", random_state=random_state, **kwargs)
    elif c == "kmedoids-erp" or c == "k-medoids-erp":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(distance="erp", random_state=random_state, **kwargs)
    elif c == "kmedoids-lcss" or c == "k-medoids-lcss":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(distance="lcss", random_state=random_state, **kwargs)
    elif c == "kmedoids-msm" or c == "k-medoids-msm":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(distance="msm", random_state=random_state, **kwargs)
    elif c == "kmedoids-twe" or c == "k-medoids-twe":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(distance="twe", random_state=random_state, **kwargs)
    elif c == "kmedoids-wdtw" or c == "k-medoids-wdtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(distance="wdtw", random_state=random_state, **kwargs)
    elif c == "kmedoids-wddtw" or c == "k-medoids-wddtw":
        from aeon.clustering import TimeSeriesKMedoids

        return TimeSeriesKMedoids(distance="wddtw", random_state=random_state, **kwargs)
