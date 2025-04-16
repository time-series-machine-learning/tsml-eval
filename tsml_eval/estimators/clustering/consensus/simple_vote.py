"""Simple vote ensemble for clustering."""

import numpy as np
import pandas as pd
from aeon.base._base import _clone_estimator
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


class SimpleVote(ClusterMixin, BaseEstimator):
    """
    SimpleVote clustering ensemble.

    SimpleVote is a clustering ensemble that uses a voting scheme to combine
    the results of multiple clusterers.

    Parameters
    ----------
    clusterers : list of clusterers, default=None
        A list of clusterers to use in the ensemble. If None, defaults to 5
        KMeans clusterers.
    n_clusters : int, default=8
        The number of clusters to form.
    random_state : int, default=None
        The seed for random number generation.

    Attributes
    ----------
    labels_ : ndarray of shape (n_instances,)
        Labels of each point from the last fit.

    Examples
    --------
    >>> from tsml_eval.estimators.clustering.consensus.simple_vote import SimpleVote
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.metrics import rand_score
    >>> iris = load_iris()
    >>> sv = SimpleVote(n_clusters=3, random_state=0)
    >>> sv.fit(iris.data)
    SimpleVote(...)
    >>> s = rand_score(iris.target, sv.labels_)
    """

    def __init__(self, clusterers=None, n_clusters=8, random_state=None):
        self.clusterers = clusterers
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit model to X using a simple vote ensemble."""
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "SimpleVote is not a time series clusterer. "
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
            if uc.shape[0] != self.n_clusters:
                raise ValueError(
                    "Input clusterers must have the same number of clusters as the "
                    f"SimpleVote n_clusters ({self.n_clusters}). Found "
                    f"{uc.shape[0]} for clusterer {i}."
                )
            elif (np.sort(uc) != np.arange(self.n_clusters)).any():
                raise ValueError(
                    "Input clusterers must have cluster labels in the range "
                    "0 to  n_clusters - 1."
                )

        self._build_ensemble(cluster_assignments)

        return self

    def predict(self, X):
        """Predict cluster labels for X."""
        rng = check_random_state(self.random_state)
        return np.array(
            [
                rng.choice(np.flatnonzero(prob == prob.max()))
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
        """Predict cluster probabilities for X."""
        check_is_fitted(self)

        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "SimpleVote is not a time series clusterer. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, reset=False)

        cluster_assignments = np.zeros(
            (len(self._clusterers), X.shape[0]), dtype=np.int32
        )

        cluster_assignments[0] = self._clusterers[0].predict(X)
        for i in range(1, len(self._clusterers)):
            cluster_assignments[i] = self._new_labels[i - 1][
                self._clusterers[i].predict(X)
            ]

        votes = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_clusters),
            axis=0,
            arr=cluster_assignments,
        ).T

        return votes / len(self._clusterers)

    def _build_ensemble(self, cluster_assignments):
        rng = check_random_state(self.random_state)

        self._new_labels = []
        new_assignments = np.zeros(
            (cluster_assignments.shape[0], cluster_assignments.shape[1]), dtype=np.int32
        )

        new_assignments[0] = cluster_assignments[0]
        for i in range(1, len(cluster_assignments)):
            cost_matrix = -np.histogram2d(
                cluster_assignments[i], cluster_assignments[0], bins=self.n_clusters
            )[0]
            _, col_indices = linear_sum_assignment(cost_matrix)
            self._new_labels.append(col_indices)
            new_assignments[i] = col_indices[cluster_assignments[i]]

        votes = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_clusters),
            axis=0,
            arr=new_assignments,
        ).T

        self.labels_ = np.array(
            [rng.choice(np.flatnonzero(v == v.max())) for v in votes]
        )
