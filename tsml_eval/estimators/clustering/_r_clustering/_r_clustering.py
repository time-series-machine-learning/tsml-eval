"""R-clustering Clusterer.

Pipeline clusterer using the R-clustering transformer and an estimator.
"""

__maintainer__ = ["chrisholder"]
__all__ = ["RClustering"]

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.clustering import BaseClusterer
from sklearn.cluster import KMeans

from tsml_eval.estimators.clustering._r_clustering._r_clustering_minirocket import (
    RClusteringTransformer,
)


class RClustering(BaseClusterer):

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": False,
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "algorithm_type": "feature",
    }

    def __init__(
        self,
        num_features: int = 500,
        max_dilations_per_kernel: int = 32,
        pca_result: bool = True,
        estimator=None,
        random_state=None,
        n_jobs=1,
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.pca_result = pca_result
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.labels_ = None

        super().__init__()

    def _fit(self, X, y=None):
        self._transformer = RClusteringTransformer(
            num_features=self.num_features,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            random_state=self.random_state,
            pca_result=self.pca_result,
        )

        self._estimator = _clone_estimator(
            (
                KMeans(random_state=self.random_state, n_init=10)
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)

        self._estimator.fit(X_t, y)

        self.labels_ = self._estimator.labels_

        return self

    def _predict(self, X) -> np.ndarray:
        return self._estimator.predict(self._transformer.transform(X))

    def _predict_proba(self, X) -> np.ndarray:
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            preds = self._estimator.predict(self._transformer.transform(X))
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

    def _score(self, X, y=None):
        raise NotImplementedError("R-clustering does not support scoring.")
