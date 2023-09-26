from math import floor

import numpy as np
from sklearn.utils.validation import check_random_state
from aeon.distances import (
    euclidean_pairwise_distance,
    dtw_pairwise_distance,
    ddtw_pairwise_distance,
    wdtw_pairwise_distance,
    wddtw_pairwise_distance,
    erp_pairwise_distance,
    lcss_pairwise_distance,
    twe_pairwise_distance,
    msm_pairwise_distance
)

DISTANCE_CANDIDATES = [
    "euclidean",
    "dtw",
    "ddtw",
    "dtw-r",
    "ddtw-r",
    "wdtw",
    "wddtw",
    "erp",
    "lcss",
    "twe",
    "msm",
]


class DistanceSplitter:
    """EE-based splitter for TS-CHIEF implementation."""

    @staticmethod
    def generate(X, y, random_state=None):
        """Generate a randomized distance splitter candidate."""
        _, dims, length = X.shape
        rng = check_random_state(random_state)

        splitter = DistanceSplitter()
        splitter.dim = rng.randint(dims)

        metric = rng.choice(DISTANCE_CANDIDATES)
        example = X[0, splitter.dim, :]
        splitter.metric = metric
        if metric == "euclidean":
            pass
        elif metric == "dtw":
            pass
        elif metric == "ddtw":
            pass
        elif metric == "dtw-r":
            max_warp = floor((length + 1) / 4)
            warp = rng.randint(0, max_warp + 1)
            splitter.window = warp / length
        elif metric == "ddtw-r":
            max_warp = floor((length + 1) / 4)
            warp = rng.randint(0, max_warp + 1)
            splitter.window = warp / length
        elif metric == "wdtw":
            splitter.g = rng.uniform(0, 1)
        elif metric == "wddtw":
            splitter.g = rng.uniform(0, 1)
        elif metric == "erp":
            sigma = np.std(X[:, splitter.dim, :])
            splitter.g = rng.uniform(sigma / 5, sigma)
        elif metric == "lcss":
            sigma = np.std(X[:, splitter.dim, :])
            splitter.epsilon = rng.uniform(sigma / 5, sigma)
            max_warp = floor((length + 1) / 4)
            splitter.window = rng.randint(0, max_warp + 1)
        elif metric == "twe":
            splitter.nu = rng.choice(
                [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            )
            splitter.lmbda = rng.choice(np.arange(0, 10) / 9)
        elif metric == "msm":
            splitter.c = rng.choice(
                [
                    0.01,
                    0.01375,
                    0.0175,
                    0.02125,
                    0.025,
                    0.02875,
                    0.0325,
                    0.03625,
                    0.04,
                    0.04375,
                    0.0475,
                    0.05125,
                    0.055,
                    0.05875,
                    0.0625,
                    0.06625,
                    0.07,
                    0.07375,
                    0.0775,
                    0.08125,
                    0.085,
                    0.08875,
                    0.0925,
                    0.09625,
                    0.1,
                    0.136,
                    0.172,
                    0.208,
                    0.244,
                    0.28,
                    0.316,
                    0.352,
                    0.388,
                    0.424,
                    0.46,
                    0.496,
                    0.532,
                    0.568,
                    0.604,
                    0.64,
                    0.676,
                    0.712,
                    0.748,
                    0.784,
                    0.82,
                    0.856,
                    0.892,
                    0.928,
                    0.964,
                    1,
                    1.36,
                    1.72,
                    2.08,
                    2.44,
                    2.8,
                    3.16,
                    3.52,
                    3.88,
                    4.24,
                    4.6,
                    4.96,
                    5.32,
                    5.68,
                    6.04,
                    6.4,
                    6.76,
                    7.12,
                    7.48,
                    7.84,
                    8.2,
                    8.56,
                    8.92,
                    9.28,
                    9.64,
                    10,
                    13.6,
                    17.2,
                    20.8,
                    24.4,
                    28,
                    31.6,
                    35.2,
                    38.8,
                    42.4,
                    46,
                    49.6,
                    53.2,
                    56.8,
                    60.4,
                    64,
                    67.6,
                    71.2,
                    74.8,
                    78.4,
                    82,
                    85.6,
                    89.2,
                    92.8,
                    96.4,
                    100,
                ]
            )

        splitter.exemplars = []
        classes = np.unique(y)
        for c in classes:
            group = np.argwhere(y == c).ravel()
            exemplar_idx = rng.choice(group)
            splitter.exemplars.append(X[exemplar_idx, splitter.dim, :])
        splitter.exemplars = np.array(splitter.exemplars)

        return splitter

    def split(self, X):
        """Split incoming data."""
        samples, _, _ = X.shape
        X = X[:, self.dim, :]
        y = self.exemplars

        if self.metric == "euclidean":
            pairwise_distances = euclidean_pairwise_distance(X, y)
        elif self.metric == "dtw":
            pairwise_distances = dtw_pairwise_distance(X, y)
        elif self.metric == "ddtw":
            pairwise_distances = ddtw_pairwise_distance(X, y)
        elif self.metric == "dtw-r":
            pairwise_distances = dtw_pairwise_distance(X, y, self.window)
        elif self.metric == "ddtw-r":
            pairwise_distances = dtw_pairwise_distance(X, y, self.window)
        elif self.metric == "wdtw":
            pairwise_distances = wdtw_pairwise_distance(X, y, g=self.g)
        elif self.metric == "wddtw":
            pairwise_distances = wddtw_pairwise_distance(X, y, g=self.g)
        elif self.metric == "erp":
            pairwise_distances = erp_pairwise_distance(X, y, g=self.g)
        elif self.metric == "lcss":
            pairwise_distances = lcss_pairwise_distance(X, y, window=self.window, epsilon=self.epsilon)
        elif self.metric == "twe":
            pairwise_distances = twe_pairwise_distance(X, y, lmbda=self.lmbda, nu=self.nu)
        else:
            pairwise_distances = msm_pairwise_distance(X, y, c=self.c)

        split_idx = np.argmin(pairwise_distances, axis=1)

        return split_idx
