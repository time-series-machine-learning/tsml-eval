"""Wrapper for K-Shape clustering algorithm."""
from aeon.clustering.base import BaseClusterer
import numpy as np

class KShapeWrapper(BaseClusterer):
    """K-Shape clustering algorithm wrapper.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    random_state : int, default=None
        Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
    params : dict, default=None
        Additional parameters to pass to the K-Shape algorithm.

    Attributes
    ----------
    cluster_centers_ : list
        List of cluster centroids after fitting.
    labels_ : np.ndarray
        Labels of each point after fitting.

    References
    ----------
    .. [1] Paparrizos, John, and Luis Gravano. "k-shape: Efficient and accurate clustering of time series." Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data. 2015.
    """

    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        super().__init__()

    def _fit(self, X, y=None):
        from kshape.core import kshape

        # Prepare data - kshape expects 3D array (n_samples, n_timepoints, 1)
        X_new = X.swapaxes(1, 2)

        # Apply k-shape clustering
        self.kshape_model = kshape(X_new, self.n_clusters, centroid_init="random",
                              max_iter=100)

    def _predict(self, X) -> np.ndarray:
        """Will only work with same X in fit."""
        # Extract predictions from k-shape result
        # kshape returns a list of (centroid, indices) tuples
        predictions = np.zeros(X.shape[0])
        for i in range(self.n_clusters):
            centroid, indices = self.kshape_model[i]
            predictions[indices] = i

        return predictions


