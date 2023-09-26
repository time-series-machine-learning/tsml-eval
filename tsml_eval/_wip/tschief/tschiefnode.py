import numpy as np
from sklearn.utils.validation import check_random_state

from tsml_eval._wip.tschief._splitters import (
    DictionarySplitter,
    DistanceSplitter,
    IntervalSplitter,
)


class TsChiefNode:
    """Tree node for the TS-CHIEF model."""

    def __init__(self, n_dictionary, n_distance, n_interval, random_state):
        self.rng = check_random_state(random_state)
        self.n_dictionary = n_dictionary
        self.n_distance = n_distance
        self.n_interval = n_interval

        self.is_leaf = False

    def fit(self, X, y, X_boss, sfas):
        """Fit the tree node with respective subset of data."""
        self._class_dtype = y.dtype

        if len(np.unique(y)) == 1:
            self.is_leaf = True
            self.label = y[0]
            return self

        best_splitter = None
        best_split = None
        best_gini = np.inf

        for i in range(self.n_dictionary):
            splitter = DictionarySplitter.generate(X_boss, y, sfas, self.rng)
            split = splitter.split_train(X_boss)
            gini = _gini(y, split)

            if gini < best_gini:
                best_splitter = splitter
                best_split = split
                best_gini = gini

        for i in range(self.n_distance):
            splitter = DistanceSplitter.generate(X, y, self.rng)
            split = splitter.split(X)
            gini = _gini(y, split)

            if gini < best_gini:
                best_splitter = splitter
                best_split = split
                best_gini = gini

        for i in range(self.n_interval):
            splitter = IntervalSplitter.generate(X, y, self.rng)
            split = splitter.split(X)
            gini = _gini(y, split)

            if gini < best_gini:
                best_splitter = splitter
                best_split = split
                best_gini = gini

        self.splitter_ = best_splitter
        self.children_ = []
        for split in np.unique(best_split):
            down_idx = np.argwhere(best_split == split).ravel()
            down_X = X[down_idx]
            down_y = y[down_idx]
            down_X_boss = np.vectorize(lambda bag, down_idx=down_idx: bag[down_idx, :])(
                X_boss
            )

            self.children_.append(
                TsChiefNode(
                    n_dictionary=self.n_dictionary,
                    n_distance=self.n_distance,
                    n_interval=self.n_interval,
                    random_state=self.rng,
                ).fit(down_X, down_y, down_X_boss, sfas)
            )

        return self

    def predict(self, X):
        """Predict class labels for subset of data."""
        samples = X.shape[0]

        if self.is_leaf:
            return np.repeat(self.label, samples)

        preds = np.empty(samples, dtype=self._class_dtype)
        split_idx = self.splitter_.split(X)
        for split in np.unique(split_idx):
            down_idx = np.argwhere(split_idx == split).ravel()
            preds[down_idx] = self.children_[split].predict(X[down_idx])

        return preds


def _gini(y, split_idx):
    ginis = []
    splits, split_sizes = np.unique(split_idx, return_counts=True)
    for split in splits:
        class_distribution = y[np.argwhere(split_idx == split).ravel()]

        _, class_counts = np.unique(class_distribution, return_counts=True)
        class_counts = class_counts / len(class_distribution)
        ginis.append(1 - np.sum(class_counts**2))

    return np.average(ginis, weights=split_sizes)
