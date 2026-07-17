"""SharedDrCIF: DrCIF features computed once, shared by a QUANT-style forest.

The "big reverse" of DrCIF's cost structure: instead of every tree paying for
its own random interval transform, the DrCIF attribute pool is computed once
over a fixed interval grid on the three DrCIF representations, and a single
extra trees ensemble (QUANT's head, with bagging and OOB train estimates) is
trained on the shared matrix. Per-tree diversity comes from bootstrap case
sampling and per-split feature subsampling instead of per-tree intervals.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["SharedDrCIF"]

import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.utils.validation import check_n_jobs

from tsml_eval._wip.classification._shared_interval_transform import (
    SharedIntervalTransform,
)


class SharedDrCIF(BaseClassifier):
    """DrCIF attribute pool over fixed intervals with a single bagged forest.

    Parameters
    ----------
    features : "drcif29" or "union35", default="drcif29"
        Per-interval attribute pool: the 29 DrCIF attributes (22 catch22 + 7
        summary stats), or those plus the six quantile features.
    interval_scheme : "dyadic" or "random", default="dyadic"
        Fixed QUANT-style dyadic grid, or a seeded random interval set of the
        same size.
    min_interval_length : int, default=3
        Minimum interval length.
    max_interval_depth : int, default=6
        Maximum dyadic depth (also sets the interval count for "random").
    max_interval_prop : float, default=0.5
        Maximum interval length as a proportion of series length for the
        "random" scheme (DrCIF's default is 0.5). Ignored for "dyadic".
    train_estimate : bool, default=False
        If True, the default ExtraTreesClassifier is built with bootstrap
        sampling and OOB scoring so fit_predict can return an out-of-bag train
        estimate (DrCIF-style). If False (default), the deployed forest trains
        each tree on all cases, matching DrCIF's and QUANT's deployment — the
        right choice for test-only runs, where bagging only costs accuracy.
        Ignored when a custom estimator is supplied.
    estimator : sklearn estimator or None, default=None
        If None, an ExtraTreesClassifier with 200 trees is used (bootstrap and
        OOB scoring controlled by train_estimate). A user-supplied estimator
        must set bootstrap=True and oob_score=True for train estimates.
    class_weight : dict, "balanced", "balanced_subsample" or None, default=None
        Only applies to the default ExtraTreesClassifier.
    random_state : int, RandomState instance or None, default=None
        Seed or RandomState for reproducibility.
    n_jobs : int, default=1
        Number of jobs for the forest fit and predict.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The unique class labels.
    fit_time_millis_ : int
        Wall clock time for the last _fit call, read by tsml-eval experiments
        so FitTime is recorded even in train-fold runs.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": False,
        "capability:multithreading": True,
        "algorithm_type": "interval",
    }

    def __init__(
        self,
        features="drcif29",
        interval_scheme="dyadic",
        min_interval_length=3,
        max_interval_depth=6,
        max_interval_prop=0.5,
        train_estimate=False,
        estimator=None,
        class_weight=None,
        random_state=None,
        n_jobs=1,
    ):
        self.features = features
        self.interval_scheme = interval_scheme
        self.min_interval_length = min_interval_length
        self.max_interval_depth = max_interval_depth
        self.max_interval_prop = max_interval_prop
        self.train_estimate = train_estimate
        self.estimator = estimator
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__()

        # only advertise train estimates when the deployed forest is bagged
        if train_estimate:
            self.set_tags(**{"capability:train_estimate": True})

    def _fit(self, X, y):
        start = time.time()
        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transformer = SharedIntervalTransform(
            features=self.features,
            interval_scheme=self.interval_scheme,
            min_interval_length=self.min_interval_length,
            max_depth=self.max_interval_depth,
            max_interval_prop=self.max_interval_prop,
            random_state=self.random_state,
        )

        self._estimator = _clone_estimator(
            (
                ExtraTreesClassifier(
                    n_estimators=200,
                    max_features=0.1,
                    criterion="entropy",
                    class_weight=self.class_weight,
                    bootstrap=self.train_estimate,
                    oob_score=self.train_estimate,
                    n_jobs=self._n_jobs,
                    random_state=self.random_state,
                )
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        X_t = self._transformer.fit_transform(X)
        self._estimator.fit(X_t, y)

        self.fit_time_millis_ = int(round((time.time() - start) * 1000))
        return self

    def _fit_predict(self, X, y) -> np.ndarray:
        probas = self._fit_predict_proba(X, y)
        return np.array([self.classes_[int(np.argmax(prob))] for prob in probas])

    def _fit_predict_proba(self, X, y) -> np.ndarray:
        self._fit(X, y)

        if not hasattr(self._estimator, "oob_decision_function_"):
            raise ValueError(
                "Train estimates require an estimator fit with bootstrap=True "
                "and oob_score=True."
            )

        probas = np.array(self._estimator.oob_decision_function_, copy=True)
        # cases never out-of-bag have nan rows, use uniform probabilities as
        # DrCIF does
        never_oob = ~np.isfinite(probas).all(axis=1)
        probas[never_oob] = 1 / self.n_classes_
        return probas

    def _predict(self, X):
        return self._estimator.predict(self._transformer.transform(X))

    def _predict_proba(self, X):
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, self._class_dictionary[preds[i]]] = 1
            return dists

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"max_interval_depth": 2}
