"""ADASYN over sampling algorithm."""

from typing import Optional, Union

import numpy as np
from sklearn import clone

from tsml_eval._wip.rt.transformations.collection.imbalance._smote import SMOTE

__maintainer__ = ["TonyBagnall"]
__all__ = ["ADASYN"]


class ADASYN(SMOTE):
    """
    Over-sampling using Adaptive Synthetic Sampling (ADASYN).

    Adaptation of imblearn.over_sampling.ADASYN
    original authors:
    #          Guillaume Lemaitre <g.lemaitre58@gmail.com>
    #          Christos Aridas
    # License: MIT

    This transformer extends SMOTE, but it generates different number of
    samples depending on an estimate of the local distribution of the class
    to be oversampled.
    """

    def __init__(
        self,
        n_neighbors=5,
        distance: Union[str, callable] = "euclidean",
        distance_params: Optional[dict] = None,
        weights: Union[str, callable] = "uniform",
        n_jobs: int = 1,
        random_state=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            distance=distance,
            distance_params=distance_params,
            weights=weights,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _transform(self, X, y=None):
        X = np.squeeze(X, axis=1)
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        self.nn_.fit(X, y)
        nn_class = clone(self.nn_)

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = X[target_class_indices]
            y_class = y[target_class_indices]

            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
            n_neighbors = self.nn_.n_neighbors - 1
            ratio_nn = np.sum(y[nns] != class_sample, axis=1) / n_neighbors
            if not np.sum(ratio_nn):
                raise RuntimeError(
                    "Not any neigbours belong to the majority"
                    " class. This case will induce a NaN case"
                    " with a division by zero. ADASYN is not"
                    " suited for this specific dataset."
                    " Use SMOTE instead."
                )
            ratio_nn /= np.sum(ratio_nn)
            n_samples_generate = np.rint(ratio_nn * n_samples).astype(int)
            n_samples = np.sum(n_samples_generate)
            if not n_samples:
                raise ValueError(
                    "No samples will be generated with the provided ratio settings."
                )

            # Reuse the cloned estimator, just refit it for each class
            nn_class.fit(X_class, y_class)
            nns = nn_class.kneighbors(X_class, return_distance=False)[:, 1:]

            enumerated_class_indices = np.arange(len(target_class_indices))
            rows = np.repeat(enumerated_class_indices, n_samples_generate)
            cols = self._random_state.choice(n_neighbors, size=n_samples)
            diffs = X_class[nns[rows, cols]] - X_class[rows]
            steps = self._random_state.uniform(size=(n_samples, 1))
            X_new = X_class[rows] + steps * diffs

            X_new = X_new.astype(X.dtype)
            y_new = np.full(n_samples, fill_value=class_sample, dtype=y.dtype)
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        X_resampled = X_resampled[:, np.newaxis, :]
        return X_resampled, y_resampled
