import numpy as np
from sklearn.utils.validation import check_random_state
from aeon.classification.base import BaseClassifier

from tsml_eval._wip.tschief._splitters import generate_boss_transforms
from tsml_eval._wip.tschief.tschiefnode import TsChiefNode

__maintainer__ = ["GuiArcencio"]
__all__ = ["TsChief"]


class TsChief(BaseClassifier):
    """Unfinished implementation of the TS-CHIEF classifier."""

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_trees=500,
        n_dictionary=100,
        n_interval=100,
        n_distance=5,
        n_boss_transformations=1000,
        random_state=None,
    ):
        self.n_trees = n_trees
        self.n_dictionary = n_dictionary
        self.n_interval = n_interval
        self.n_distance = n_distance
        self.n_boss_transformations = n_boss_transformations
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        self._class_dtype = y.dtype
        self.random_state = check_random_state(self.random_state)

        sfas, X_boss = generate_boss_transforms(
            X, self.n_boss_transformations, self.random_state
        )
        self.trees_ = []
        for _ in range(self.n_trees):
            self.trees_.append(
                TsChiefNode(
                    n_dictionary=self.n_dictionary,
                    n_distance=self.n_distance,
                    n_interval=self.n_interval,
                    random_state=self.random_state,
                ).fit(X, y, X_boss, sfas)
            )

        return self

    def _predict(self, X):
        samples = X.shape[0]

        votes = np.empty((samples, self.n_trees), dtype=self._class_dtype)
        preds = np.empty(samples, dtype=self._class_dtype)

        for i, tree in enumerate(self.trees_):
            votes[:, i] = tree.predict(X)

        for i in range(samples):
            classes, counts = np.unique(votes[i], return_counts=True)
            max_idx = np.argmax(counts)
            preds[i] = classes[max_idx]

        return preds
