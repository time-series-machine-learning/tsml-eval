import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from tsml.base import _clone_estimator


class IterativeVotingClustering(BaseEstimator, ClusterMixin):
    """
    IVC (Iterative Voting Clustering) Consensus Clusterer.

    IVC is a consensus clustering algorithm that combines the results of multiple
    base clusterers to find a consensus clustering. It iteratively refines cluster
    assignments based on a majority voting scheme.

    Parameters:
    -----------
    clusterers : list
        List of base clusterers to be used for consensus clustering.
    num_clusters : int, optional (default=2)
        The number of clusters to generate.
    max_iterations : int, optional (default=200)
        Maximum number of iterations for the consensus clustering.
    seed_clusterer : bool, optional (default=False)
        Whether to use a fixed random seed for cluster initialization.

    """

    def __init__(self, clusterers=None, n_clusters=8, max_iterations=500,
                 random_state=None):
        self.clusterers = clusterers
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state


    def fit(self, X, y=None):
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "SimpleVote is not a time series classifier. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, ensure_min_samples=self.n_clusters)

        rng = check_random_state(self.random_state)

        if self.clusterers is None:
            self._clusterers = [KMeans(n_clusters=self.n_clusters, n_init="auto", random_state=rng.randint(np.iinfo(np.int32).max)) for _ in range(5)]
        else:
            self._clusterers = [_clone_estimator(clusterer, random_state=rng.randint(np.iinfo(np.int32).max)) for clusterer in self.clusterers]

        cluster_assignments = np.zeros((len(self._clusterers), X.shape[0]), dtype=np.int32)
        for i, clusterer in enumerate(self._clusterers):
            clusterer.fit(X)
            cluster_assignments[i] = clusterer.labels_

            uc = np.unique(clusterer.labels_)
            if (np.sort(uc) != np.arange(self.n_clusters)).any():
                raise ValueError(
                    "Input clusterers must have cluster labels in the range "
                    "0 to  n_clusters - 1."
                )

        self._build_ensemble(cluster_assignments)

        return self

    def predict(self, X):
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "SimpleVote is not a time series classifier. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, reset=False)

        cluster_assignments = np.zeros((len(self._clusterers), X.shape[0]), dtype=np.int32)
        for i in range(0, len(self._clusterers)):
            cluster_assignments[i] = self._clusterers[i].predict(X)

        rng = check_random_state(self.random_state)
        labels = self._calculate_cluster_membership(cluster_assignments, rng)

        return labels

    def _build_ensemble(self, cluster_assignments):
        rng = check_random_state(self.random_state)
        self.labels_ = rng.randint(self.n_clusters, size=cluster_assignments.shape[1])

        iterations = 0
        while iterations < self.max_iterations:
            self._select_cluster_centers(cluster_assignments, rng)
            labels = self._calculate_cluster_membership(cluster_assignments, rng)

            unique = np.unique(labels)
            if unique.shape[0] != self.n_clusters:
                for i in range(self.n_clusters):
                    if i not in unique:
                        x = np.concatenate([np.where(labels == unique[i])[0] for i in
                                            range(unique.shape[0])])
                        labels[rng.choice(x)] = i

            if (labels == self.labels_).all():
                break

            self.labels_ = labels
            iterations += 1

    def _select_cluster_centers(self, cluster_assignments, rng):
        self._cluster_centers = np.zeros((self.n_clusters, cluster_assignments.shape[0]), dtype=np.int32)

        for i in range(self.n_clusters):
            cluster_indices = np.where(self.labels_ == i)[0]
            p = cluster_assignments[:, cluster_indices]

            for n, att in enumerate(p):
                unique, counts = np.unique(att, return_counts=True)
                self._cluster_centers[i][n] = unique[rng.choice(np.flatnonzero(counts == counts.max()))]

    def _calculate_cluster_membership(self, cluster_assignments, rng):
        labels = np.zeros(cluster_assignments.shape[1], dtype=np.int32)

        for i in range(cluster_assignments.shape[1]):
            dists = np.sum(cluster_assignments[:, i] != self._cluster_centers, axis=1)

            min_indices = np.where(dists == dists.min())[0]

            if len(min_indices) > 1:
                labels[i] = rng.choice(min_indices)
            else:
                labels[i] = min_indices[0]

        return labels
