"""SharedDrCIFRegressor: DrCIF features computed once, shared by a forest.

The regression counterpart of ``SharedDrCIF``. Same "big reverse" of DrCIF's
cost structure: instead of every tree paying for its own random interval
transform, the DrCIF attribute pool is computed once over a fixed interval grid
on the three DrCIF representations, and a single extra trees ensemble (with
optional bagging and OOB train estimates) is trained on the shared matrix.
Per-tree diversity comes from bootstrap case sampling and per-split feature
subsampling instead of per-tree intervals.

The interval feature transform (``SharedIntervalTransform``) is shared with the
classifier — it is representation/target agnostic — so only the estimator head
and the aeon base class differ here.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["SharedDrCIFRegressor"]

import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from aeon.base._base import _clone_estimator
from aeon.regression.base import BaseRegressor
from aeon.utils.validation import check_n_jobs

from tsml_eval._wip.classification._shared_interval_transform import (
    SharedIntervalTransform,
)


class SharedDrCIFRegressor(BaseRegressor):
    """DrCIF attribute pool over fixed intervals with a single bagged forest.

    Parameters
    ----------
    features : "drcif29", "union35" or "minimal", default="drcif29"
        Per-interval attribute pool. "drcif29" = 22 catch22 + 7 summary stats;
        "union35" = those plus 6 quantile features; "minimal" = the data-driven
        lean pool of 14 (5 cheap high-importance catch22 + 7 summary + 2 PULSAR
        order-aware stats), dropping the expensive low-value catch22 tail.
    interval_scheme : "dyadic" or "random", default="random"
        Seeded random interval set using DrCIF's generation rule, or the fixed
        QUANT-style dyadic grid of the same size.
    min_interval_length : int, default=3
        Minimum interval length.
    max_interval_depth : int, default=6
        Maximum dyadic depth (also sets the interval count for "random").
    max_interval_prop : float, default=0.5
        Maximum interval length as a proportion of series length for the
        "random" scheme (DrCIF's default is 0.5). Ignored for "dyadic".
    banded : bool, default=False
        If True, each interval computes only the catch22 features whose length
        threshold it clears (length-gated feature scaling); summary and
        quantile stats are always computed. Cuts cost and avoids degenerate
        catch22 values on short intervals.
    dilation : bool, default=False
        Only used with interval_scheme="random". If True, each interval draws a
        random dilation scaled to its length (geometrically decaying towards
        d=1), expanding its window for a multi-scale view at no extra feature
        cost.
    drop_constant : bool, default=True
        If True, drop zero-variance (constant, e.g. all-zero) feature columns on
        the training set before building the forest, and again at predict time.
    tree_type : "extra" or "dt", default="extra"
        "extra" uses a single ExtraTreesRegressor (random split thresholds,
        per-split feature sampling). "dt" uses a random-subspace ensemble of
        proper best-split DecisionTreeRegressors, matching DrCIF's tree
        structure. Ignored if estimator is set.
    n_estimators : int, default=200
        Number of trees in the default ensemble.
    max_features : float, default=0.1
        Feature sampling fraction: per-split for "extra", per-tree (random
        subspace) for "dt".
    train_estimate : bool, default=False
        If True, the default ExtraTreesRegressor is built with bootstrap
        sampling and OOB scoring so fit_predict can return an out-of-bag train
        estimate (DrCIF-style). If False (default), each tree trains on all
        cases, matching DrCIF's/QUANT's deployment. Ignored when a custom
        estimator is supplied.
    estimator : sklearn regressor or None, default=None
        If None, an ExtraTreesRegressor with 200 trees is used (bootstrap and
        OOB scoring controlled by train_estimate). A user-supplied estimator
        must set bootstrap=True and oob_score=True for train estimates.
    random_state : int, RandomState instance or None, default=None
        Seed or RandomState for reproducibility.
    n_jobs : int, default=1
        Number of jobs for the forest fit and predict.

    Attributes
    ----------
    fit_time_millis_ : int
        Wall clock time for the last _fit call, read by tsml-eval experiments
        so FitTime is recorded even in train-fold runs.
    n_features_used_ : int
        Number of feature columns kept after the constant-column filter.
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
        interval_scheme="random",
        min_interval_length=3,
        max_interval_depth=6,
        max_interval_prop=0.5,
        banded=False,
        dilation=False,
        representations=("base", "diff1", "periodogram"),
        fixed_lengths=(9, 16, 32),
        drop_constant=True,
        tree_type="extra",
        n_estimators=200,
        max_features=0.1,
        train_estimate=False,
        estimator=None,
        random_state=None,
        n_jobs=1,
    ):
        self.features = features
        self.interval_scheme = interval_scheme
        self.min_interval_length = min_interval_length
        self.max_interval_depth = max_interval_depth
        self.max_interval_prop = max_interval_prop
        self.banded = banded
        self.dilation = dilation
        self.representations = representations
        self.fixed_lengths = fixed_lengths
        self.drop_constant = drop_constant
        self.tree_type = tree_type
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.train_estimate = train_estimate
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__()

        # only advertise train estimates when the deployed forest is bagged
        if train_estimate:
            self.set_tags(**{"capability:train_estimate": True})

    def _default_estimator(self):
        """Build the default regressor per tree_type.

        "extra": a single ExtraTreesRegressor (random split thresholds,
        per-split feature sampling). "dt": a random-subspace ensemble of proper
        best-split DecisionTreeRegressors, matching DrCIF's tree structure.
        """
        if self.tree_type == "extra":
            return ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                criterion="squared_error",
                bootstrap=self.train_estimate,
                oob_score=self.train_estimate,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
        elif self.tree_type == "dt":
            from sklearn.ensemble import BaggingRegressor
            from sklearn.tree import DecisionTreeRegressor

            return BaggingRegressor(
                estimator=DecisionTreeRegressor(criterion="squared_error"),
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                bootstrap=self.train_estimate,
                bootstrap_features=False,
                oob_score=self.train_estimate,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown tree_type: {self.tree_type}")

    def _fit(self, X, y):
        start = time.time()
        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transformer = SharedIntervalTransform(
            features=self.features,
            interval_scheme=self.interval_scheme,
            min_interval_length=self.min_interval_length,
            max_depth=self.max_interval_depth,
            max_interval_prop=self.max_interval_prop,
            banded=self.banded,
            dilation=self.dilation,
            representations=self.representations,
            fixed_lengths=self.fixed_lengths,
            random_state=self.random_state,
        )

        self._estimator = _clone_estimator(
            self.estimator if self.estimator is not None
            else self._default_estimator(),
            self.random_state,
        )

        X_t = self._transformer.fit_transform(X)

        # drop zero-variance (constant / all-zero) columns: no information, but
        # they still count toward the forest's max_features sampling
        if self.drop_constant:
            self._keep_cols_ = X_t.std(axis=0) > 0
            if not self._keep_cols_.any():  # degenerate guard
                self._keep_cols_ = np.ones(X_t.shape[1], dtype=bool)
        else:
            self._keep_cols_ = np.ones(X_t.shape[1], dtype=bool)
        self.n_features_used_ = int(self._keep_cols_.sum())

        self._estimator.fit(X_t[:, self._keep_cols_], y)

        self.fit_time_millis_ = int(round((time.time() - start) * 1000))
        return self

    def _transform_kept(self, X):
        """Transform then keep only the columns retained at fit time."""
        return self._transformer.transform(X)[:, self._keep_cols_]

    def _fit_predict(self, X, y) -> np.ndarray:
        self._fit(X, y)

        if not hasattr(self._estimator, "oob_prediction_"):
            raise ValueError(
                "Train estimates require an estimator fit with bootstrap=True "
                "and oob_score=True."
            )

        preds = np.array(self._estimator.oob_prediction_, copy=True)
        # cases never out-of-bag have nan predictions, fall back to the mean of
        # the training targets
        never_oob = ~np.isfinite(preds)
        if never_oob.any():
            preds[never_oob] = float(np.mean(y))
        return preds

    def _predict(self, X) -> np.ndarray:
        return self._estimator.predict(self._transform_kept(X))

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"max_interval_depth": 2}
