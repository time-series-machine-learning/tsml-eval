"""IVC consensus clustering algorithm."""

import numpy as np
import pandas as pd
from aeon.base._base import _clone_estimator
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


class IterativeVotingClustering(ClusterMixin, BaseEstimator):
    """
    IVC (Iterative Voting Clustering) Consensus Clusterer.

    IVC is a consensus clustering algorithm that combines the results of multiple
    base clusterers to find a consensus clustering. It iteratively refines cluster
    assignments based on a majority voting scheme.

    Parameters
    ----------
    clusterers : list of clusterers, default=None
        A list of clusterers to use in the ensemble. If None, defaults to 5
        KMeans clusterers.
    init : {'plus', 'random', 'aligned'}, default='plus'
        The method used to initialize the cluster centers. 'plus' uses the
        k-means++ initialization method, 'random' uses random selection, and
        'aligned' uses a method that aligns the cluster assignmeds from the base
        clusterers.
    n_clusters : int, default=8
        The number of clusters to form.
    max_iterations : int, default=500
        The maximum number of iterations to perform.
    random_state : int, default=None
        The seed for random number generation.

    Attributes
    ----------
    labels_ : ndarray of shape (n_instances,)
        Labels of each point from the last fit.

    Examples
    --------
    >>> from tsml_eval.estimators.clustering.consensus.ivc import (
    ...     IterativeVotingClustering
    ... )
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.metrics import rand_score
    >>> iris = load_iris()
    >>> ivc = IterativeVotingClustering(n_clusters=3, random_state=0)
    >>> ivc.fit(iris.data)
    IterativeVotingClustering(...)
    >>> s = rand_score(iris.target, ivc.labels_)
    """

    def __init__(
        self,
        clusterers=None,
        init="plus",
        n_clusters=8,
        max_iterations=500,
        random_state=None,
    ):
        self.clusterers = clusterers
        self.init = init
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit model to X using IVC."""
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "IterativeVotingClustering is not a time series clusterer. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, ensure_min_samples=self.n_clusters)

        rng = check_random_state(self.random_state)

        if self.clusterers is None:
            self._clusterers = [
                KMeans(
                    n_clusters=self.n_clusters,
                    n_init="auto",
                    random_state=rng.randint(np.iinfo(np.int32).max),
                )
                for _ in range(5)
            ]
        else:
            self._clusterers = [
                _clone_estimator(
                    clusterer, random_state=rng.randint(np.iinfo(np.int32).max)
                )
                for clusterer in self.clusterers
            ]

        cluster_assignments = np.zeros(
            (len(self._clusterers), X.shape[0]), dtype=np.int32
        )
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
        """Predict cluster labels for X."""
        check_is_fitted(self)

        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "IterativeVotingClustering is not a time series clusterer. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, reset=False)

        cluster_assignments = np.zeros(
            (len(self._clusterers), X.shape[0]), dtype=np.int32
        )
        for i in range(0, len(self._clusterers)):
            cluster_assignments[i] = self._clusterers[i].predict(X)

        rng = check_random_state(self.random_state)
        labels = self._calculate_cluster_membership(cluster_assignments, rng)

        return labels

    def _build_ensemble(self, cluster_assignments):
        rng = check_random_state(self.random_state)

        if self.init == "plus":
            self._initial_cluster_centers_plus(cluster_assignments, rng)
        elif self.init == "random":
            self._initial_cluster_centers_random(cluster_assignments, rng)
        elif self.init == "aligned":
            self._initial_cluster_centers_aligned(cluster_assignments, rng)
        else:
            raise ValueError("Invalid init method")

        self.labels_ = np.full(cluster_assignments.shape[1], -1, dtype=np.int32)

        iterations = 0
        while iterations < self.max_iterations:
            if iterations > 0:
                self._select_cluster_centers(cluster_assignments, rng)

            labels = self._calculate_cluster_membership(cluster_assignments, rng)
            labels = self._ensure_all_clusters_in_labels(labels, rng)

            if (labels == self.labels_).all():
                break

            self.labels_ = labels
            iterations += 1

    def _initial_cluster_centers_plus(self, cluster_assignments, rng):
        cluster_assignments = cluster_assignments.T

        self._cluster_centers = np.zeros(
            (self.n_clusters, cluster_assignments.shape[1]), dtype=np.int32
        )
        self._cluster_centers[0] = cluster_assignments[
            rng.randint(cluster_assignments.shape[0])
        ]

        for i in range(1, self.n_clusters):
            dists = np.sum(cluster_assignments != self._cluster_centers[i - 1], axis=1)

            probability = dists / dists.sum()
            next_center_index = rng.choice(cluster_assignments.shape[0], p=probability)

            self._cluster_centers[i] = cluster_assignments[next_center_index]

    def _initial_cluster_centers_random(self, cluster_assignments, rng):
        self.labels_ = rng.randint(self.n_clusters, size=cluster_assignments.shape[1])
        self.labels_ = self._ensure_all_clusters_in_labels(self.labels_, rng)

        self._select_cluster_centers(cluster_assignments, rng)

    def _initial_cluster_centers_aligned(self, cluster_assignments, rng):
        new_assignments = np.zeros(
            (cluster_assignments.shape[0], cluster_assignments.shape[1]), dtype=np.int32
        )

        new_assignments[0] = cluster_assignments[0]
        for i in range(1, len(cluster_assignments)):
            cost_matrix = -np.histogram2d(
                cluster_assignments[i], cluster_assignments[0], bins=self.n_clusters
            )[0]
            _, col_indices = linear_sum_assignment(cost_matrix)
            new_assignments[i] = col_indices[cluster_assignments[i]]

        votes = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_clusters),
            axis=0,
            arr=new_assignments,
        ).T

        self.labels_ = np.array(
            [rng.choice(np.flatnonzero(v == v.max())) for v in votes]
        )
        self.labels_ = self._ensure_all_clusters_in_labels(self.labels_, rng)

        self._select_cluster_centers(cluster_assignments, rng)

    def _select_cluster_centers(self, cluster_assignments, rng):
        self._cluster_centers = np.zeros(
            (self.n_clusters, cluster_assignments.shape[0]), dtype=np.int32
        )

        for i in range(self.n_clusters):
            cluster_indices = np.where(self.labels_ == i)[0]
            p = cluster_assignments[:, cluster_indices]

            for n, att in enumerate(p):
                unique, counts = np.unique(att, return_counts=True)
                self._cluster_centers[i][n] = unique[
                    rng.choice(np.flatnonzero(counts == counts.max()))
                ]

        for i in range(1, self.n_clusters):
            for j in range(i):
                if (self._cluster_centers[i] == self._cluster_centers[j]).all():
                    self._cluster_centers[i] = cluster_assignments[
                        :, rng.randint(cluster_assignments.shape[1])
                    ]

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

    def _ensure_all_clusters_in_labels(self, labels, rng):
        unique = np.unique(labels)
        if unique.shape[0] != self.n_clusters:
            for i in range(self.n_clusters):
                if i not in unique:
                    x = np.concatenate(
                        [
                            np.where(labels == unique[i])[0]
                            for i in range(unique.shape[0])
                        ]
                    )
                    labels[rng.choice(x)] = i

        return labels
