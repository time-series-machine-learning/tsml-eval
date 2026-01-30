from aeon.clustering import (
    KASBA,
    BaseClusterer,
    KSpectralCentroid,
    TimeSeriesKMeans,
    TimeSeriesKMedoids,
    TimeSeriesKShape,
)


def kasba_clusterer(n_clusters: int, random_state: int, n_jobs: int) -> BaseClusterer:
    return KASBA(
        n_clusters=n_clusters,
        distance="msm",
        ba_subset_size=0.5,
        initial_step_size=0.05,
        max_iter=300,
        tol=1e-6,
        distance_params={"c": 1.0},
        decay_rate=0.1,
        verbose=False,
        random_state=random_state,
    )


def kasba_clusterer_vldb(
    n_clusters: int, random_state: int, n_jobs: int
) -> BaseClusterer:
    return KASBA(
        n_clusters=n_clusters,
        distance="msm",
        ba_subset_size=0.5,
        initial_step_size=0.05,
        max_iter=100,
        tol=1e-6,
        distance_params={"c": 1.0},
        decay_rate=0.1,
        verbose=False,
        random_state=random_state,
    )


def kasba_twe_clusterer(
    n_clusters: int, random_state: int, n_jobs: int
) -> BaseClusterer:
    return KASBA(
        n_clusters=n_clusters,
        distance="twe",
        ba_subset_size=0.5,
        initial_step_size=0.05,
        max_iter=300,
        tol=1e-6,
        distance_params={
            "lmbda": 0.01,
            "nu": 2.0,
        },
        decay_rate=0.1,
        verbose=False,
        random_state=random_state,
    )


def dba_clusterer(n_clusters: int, random_state: int, n_jobs: int) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="dtw",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        averaging_method="ba",
        distance_params=None,
        average_params=None,
        n_jobs=n_jobs,
    )


def shape_dba_clusterer(
    n_clusters: int, random_state: int, n_jobs: int
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="shape_dtw",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        averaging_method="ba",
        distance_params={"reach": 15},
        average_params={"reach": 15},
        n_jobs=n_jobs,
    )


def mba_clusterer(n_clusters: int, random_state: int, n_jobs: int) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="msm",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        averaging_method="ba",
        distance_params={"c": 1.0},
        average_params={"c": 1.0},
        n_jobs=n_jobs,
    )


def soft_dba_clusterer(
    n_clusters: int, random_state: int, n_jobs: int
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="soft_dtw",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        averaging_method="soft",
        distance_params={"gamma": 1.0},
        average_params={"gamma": 1.0},
        n_jobs=n_jobs,
    )


def euclid_clusterer(n_clusters: int, random_state: int, n_jobs: int) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="euclidean",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        averaging_method="mean",
        distance_params=None,
        average_params=None,
        n_jobs=n_jobs,
    )


def euclid_clusterer_vldb(
    n_clusters: int, random_state: int, n_jobs: int
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="random",
        distance="euclidean",
        n_init=1,
        max_iter=100,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        averaging_method="mean",
        distance_params=None,
        average_params=None,
        n_jobs=n_jobs,
    )


def msm_clusterer(n_clusters: int, random_state: int, n_jobs: int) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="msm",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        averaging_method="mean",
        distance_params={"c": 1.0},
        average_params=None,  # This isn't used when mean selected
        n_jobs=n_jobs,
    )


def k_sc_clusterer(n_clusters: int, random_state: int, n_jobs: int) -> BaseClusterer:
    return KSpectralCentroid(
        n_clusters=n_clusters,
        max_shift=None,  # This means it will be calculated automatically to length m
        init="kmeans++",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def pam_clusterer(n_clusters: int, random_state: int, n_jobs: int) -> BaseClusterer:
    return TimeSeriesKMedoids(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="msm",
        method="pam",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        distance_params={
            "c": 1.0,
        },
        n_jobs=n_jobs,
    )


def kshape_clusterer(n_clusters: int, random_state: int, n_jobs: int) -> BaseClusterer:
    return TimeSeriesKShape(
        n_clusters=n_clusters,
        centroid_init="kmeans++",
        max_iter=300,
        n_init=1,
        random_state=random_state,
        verbose=False,
        tol=1e-6,
    )


def kshape_clusterer_vldb(
    n_clusters: int, random_state: int, n_jobs: int
) -> BaseClusterer:
    return TimeSeriesKShape(
        n_clusters=n_clusters,
        centroid_init="random",
        max_iter=100,
        n_init=1,
        random_state=random_state,
        verbose=True,
        tol=1e-6,
    )


EXPERIMENT_MODELS = {
    "KASBA": kasba_clusterer,
    "DBA": dba_clusterer,
    "shape-DBA": shape_dba_clusterer,
    "soft-DBA": soft_dba_clusterer,
    "MBA": mba_clusterer,
    "Euclid": euclid_clusterer,
    "MSM": msm_clusterer,
    "k-Shape": kshape_clusterer,
    "k-SC": k_sc_clusterer,
    "PAM-MSM": pam_clusterer,
    "KASBA-twe": kasba_twe_clusterer,
    "KASBA-vldb": kasba_clusterer_vldb,
    "Euclid-vldb": euclid_clusterer_vldb,
    "k-Shape-vldb": kshape_clusterer_vldb,
}
