# -*- coding: utf-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_random_state
from aeon.classification.interval_based._rise import (
    _round_to_nearest_power_of_two,
    acf,
    ps,
)


def _acf(X, istart, iend, lag):
    n_instances, _ = X.shape
    acf_x = np.empty(shape=(n_instances, lag))
    for j in range(n_instances):
        interval_x = X[j, istart:iend]
        acf_x[j] = acf(interval_x, lag)

    return acf_x


def _ps(X, istart, iend, lag):
    n_instances, _ = X.shape
    ps_len = _round_to_nearest_power_of_two(istart - iend)
    ps_x = np.empty(shape=(n_instances, ps_len))
    for j in range(n_instances):
        interval_x = X[j, istart:iend]
        ps_x[j] = ps(interval_x, n=ps_len * 2)

    return ps_x


FEATURE_CANDIDATES = [_acf, _ps]


class IntervalSplitter:
    """RISE-based splitter for TS-CHIEF implementation."""

    @staticmethod
    def generate(X, y, random_state=None):
        """Generate a randomized interval splitter candidate."""
        samples, dims, length = X.shape
        splitter = IntervalSplitter()
        splitter.rng = check_random_state(random_state)
        splitter.dim = splitter.rng.randint(dims)

        min_interval = min(16, length)
        acf_min_values = 4

        splitter.istart = splitter.rng.randint(0, length - min_interval)
        splitter.iend = splitter.rng.randint(splitter.istart + min_interval, length)
        acf_lag = 100
        if acf_lag > splitter.iend - splitter.istart - acf_min_values:
            acf_lag = splitter.iend - splitter.istart - acf_min_values
        if acf_lag < 0:
            acf_lag = 1

        splitter.acf_lag = acf_lag

        splitter.transform = splitter.rng.choice(FEATURE_CANDIDATES)
        X_transformed = splitter.transform(
            X[:, splitter.dim, :], splitter.istart, splitter.iend, splitter.acf_lag
        )

        splitter.tree = DecisionTreeClassifier(
            criterion="gini", max_depth=1, max_features=None, random_state=splitter.rng
        ).fit(X_transformed, y)

        return splitter

    def split(self, X):
        """Split incoming data."""
        X = X[:, self.dim, :]
        X_transformed = self.transform(X, self.istart, self.iend, self.acf_lag)

        return self.tree.apply(X_transformed) - 1
