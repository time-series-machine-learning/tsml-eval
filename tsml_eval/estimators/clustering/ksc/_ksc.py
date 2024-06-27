"""KSC implementation"""
from typing import Union, Callable
import numpy as np
from numpy.random import RandomState
from aeon.clustering import TimeSeriesKMeans
from tsml_eval.estimators.clustering.ksc._shift_invariant_distance import (
    shift_invariant_pairwise_distance, shift_invariant_distance
)
from tsml_eval.estimators.clustering.ksc._shift_invariant_average import (
    shift_invariant_average
)


class KSC(TimeSeriesKMeans):

    def __init__(
            self,
            n_clusters: int = 8,
            max_shift: Union[int, None] = None,
            init_algorithm: Union[str, np.ndarray] = "random",
            n_init: int = 10,
            max_iter: int = 300,
            tol: float = 1e-6,
            verbose: bool = False,
            random_state: Union[int, RandomState] = None,
    ):
        self.max_shift = max_shift
        self._max_shift = max_shift

        super().__init__(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            distance=shift_invariant_distance,
            distance_params={
                "max_shift": max_shift
            }
        )

    def _check_params(self, X: np.ndarray) -> None:
        if self._max_shift is None:
            # set max_shift to the length of the time series
            self._max_shift = X.shape[-1]
        super()._check_params(X)

    def _fit_one_init(self, X: np.ndarray) -> tuple:
        if isinstance(self._init_algorithm, Callable):
            cluster_centres = self._init_algorithm(X)
        else:
            cluster_centres = self._init_algorithm
        prev_inertia = np.inf
        prev_labels = None

        for i in range(self.max_iter):
            curr_pw = shift_invariant_pairwise_distance(
                cluster_centres, X, max_shift=self._max_shift
            )
            curr_labels = curr_pw.argmin(axis=0)
            curr_inertia = curr_pw.min(axis=0).sum()

            # If an empty cluster is encountered
            if np.unique(curr_labels).size < self.n_clusters:
                curr_pw, curr_labels, curr_inertia, cluster_centres = (
                    self._handle_empty_cluster(
                        X, cluster_centres, curr_pw, curr_labels, curr_inertia
                    )
                )
            if prev_labels is not None and np.array_equal(curr_labels, prev_labels):
                break
            if self.verbose:
                print("%.3f" % curr_inertia, end=" --> ")  # noqa: T001, T201

            change_in_centres = np.abs(prev_inertia - curr_inertia)
            prev_inertia = curr_inertia

            prev_labels = curr_labels

            if change_in_centres < self.tol:
                break

            # Compute new cluster centres
            for j in range(self.n_clusters):
                cluster_centres[j] = shift_invariant_average(
                    X[curr_labels == j],
                    initial_center=cluster_centres[j],
                    max_shift=self._max_shift
                )

            if self.verbose is True:
                print(f"Iteration {i}, inertia {prev_inertia}.")  # noqa: T001, T201

        return prev_labels, cluster_centres, prev_inertia, i + 1

    def _predict(self, X: np.ndarray, y=None) -> np.ndarray:
        pairwise_matrix = shift_invariant_pairwise_distance(
            self.cluster_centers_,
            X,
            max_shift=self._max_shift
        )
        return pairwise_matrix.argmin(axis=0)

    def predict_proba(self, X):
        """Predict cluster probabilities for X."""
        preds = self.predict(X)
        unique = np.unique(preds)
        for i, u in enumerate(unique):
            preds[preds == u] = i
        n_cases = len(preds)
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = int(max(preds)) + 1
        dists = np.zeros((X.shape[0], n_clusters))
        for i in range(n_cases):
            dists[i, preds[i]] = 1
        return dists
