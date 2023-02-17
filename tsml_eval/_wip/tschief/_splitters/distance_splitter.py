# -*- coding: utf-8 -*-
from math import floor

import numpy as np
from sklearn.utils.validation import check_random_state
from sktime.distances import distance_factory

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
    @staticmethod
    def generate(X, y, random_state=None):
        _, dims, length = X.shape
        rng = check_random_state(random_state)

        splitter = DistanceSplitter()
        splitter.dim = rng.randint(dims)

        metric = rng.choice(DISTANCE_CANDIDATES)
        example = X[0, splitter.dim, :]
        if metric == "euclidean":
            splitter.distance = distance_factory(example, example, "euclidean")
        elif metric == "dtw":
            splitter.distance = distance_factory(example, example, "dtw")
        elif metric == "ddtw":
            splitter.distance = distance_factory(example, example, "ddtw")
        elif metric == "dtw-r":
            max_warp = floor((length + 1) / 4)
            warp = rng.randint(0, max_warp + 1)
            splitter.distance = distance_factory(
                example, example, "dtw", kwargs={"window": warp / length}
            )
        elif metric == "ddtw-r":
            max_warp = floor((length + 1) / 4)
            warp = rng.randint(0, max_warp + 1)
            splitter.distance = distance_factory(
                example, example, "ddtw", kwargs={"window": warp / length}
            )
        elif metric == "wdtw":
            g = rng.uniform(0, 1)
            splitter.distance = distance_factory(
                example, example, "wdtw", kwargs={"g": g}
            )
        elif metric == "wddtw":
            g = rng.uniform(0, 1)
            splitter.distance = distance_factory(
                example, example, "wddtw", kwargs={"g": g}
            )
        elif metric == "erp":
            sigma = np.std(X[:, splitter.dim, :])
            g = rng.uniform(sigma / 5, sigma)
            splitter.distance = distance_factory(
                example, example, "erp", kwargs={"g": g}
            )
        elif metric == "lcss":
            sigma = np.std(X[:, splitter.dim, :])
            epsilon = rng.uniform(sigma / 5, sigma)
            max_warp = floor((length + 1) / 4)
            warp = rng.randint(0, max_warp + 1)
            splitter.distance = distance_factory(
                example, example, "lcss", kwargs={"epsilon": epsilon, "window": warp}
            )
        elif metric == "twe":
            nu = rng.choice(
                [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            )
            lmbda = rng.choice(np.arange(0, 10) / 9)
            splitter.distance = distance_factory(
                example, example, "twe", kwargs={"lmbda": lmbda, "nu": nu}
            )
        elif metric == "msm":
            c = rng.choice(
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
            splitter.distance = distance_factory(
                example, example, "msm", kwargs={"c": c}
            )

        splitter.exemplars = []
        classes = np.unique(y)
        for c in classes:
            group = np.argwhere(y == c).ravel()
            exemplar_idx = rng.choice(group)
            splitter.exemplars.append(X[exemplar_idx, splitter.dim, :])

        return splitter

    def split(self, X):
        samples, _, _ = X.shape

        split_idx = np.empty(samples, dtype=int)
        for i in range(samples):
            distances = [
                self.distance(X[i, self.dim, :], exemplar)
                for exemplar in self.exemplars
            ]
            split_idx[i] = np.argmin(distances)

        return split_idx
