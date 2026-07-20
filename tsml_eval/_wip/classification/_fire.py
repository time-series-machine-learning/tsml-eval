"""FIRE: Fast Interval Representation Ensemble.

The consolidated successor to DrCIF. It keeps DrCIF's three representations
(base series, first differences, periodogram) and its random interval-length
model, but replaces the per-tree interval forest with a shared-transform
architecture and an extensible ensemble of classifier heads.

Transform (computed ONCE, not per tree):
  * random-length intervals drawn once, with dilation (stride sampling scaled to
    interval length) expanding the receptive field at no extra feature cost;
  * length-gating: short intervals skip catch22 features degenerate at that
    length;
  * a lean 14-feature pool (7 summary stats + 2 order-aware PULSAR stats + the 5
    cheap high-importance catch22 features);
  * a constant-feature filter before the heads.

Heads (averaged): an ExtraTreesClassifier and a scaled Ridge
(StandardScaler + RidgeClassifierCV, decision function softmax-converted to
probabilities). The PULSAR ablation showed the Ridge/ExtraTrees pair adds ~0.8%
over ExtraTrees alone. The ``heads`` list is the extension point: further heads
can be appended and are averaged with equal weight.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["FIRE"]

import time

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.base._base import _clone_estimator
from aeon.utils.validation import check_n_jobs

from tsml_eval._wip.classification._shared_drcif import SharedDrCIF
from tsml_eval._wip.classification._shared_interval_transform import (
    SharedIntervalTransform,
)

_VALID_HEADS = ("extratrees", "ridge")


class FIRE(SharedDrCIF):
    """Fast Interval Representation Ensemble.

    Random dilated shared intervals on three representations, a lean feature
    pool, length gating and a constant-feature filter, classified by an
    averaged ensemble of heads (ExtraTrees + scaled Ridge by default). The
    settled architecture is fixed; the parameters below remain tunable. See
    SharedDrCIF for the transform parameter and attribute documentation.

    Parameters
    ----------
    features : "minimal", "drcif29" or "union35", default="minimal"
        Per-interval attribute pool. "minimal" is FIRE's lean 14-feature pool.
    heads : tuple of str, default=("extratrees", "ridge")
        Classifier heads to average, from {"extratrees", "ridge"}. Extension
        point for growing the ensemble.
    min_interval_length : int, default=3
        Minimum interval point count.
    max_interval_depth : int, default=6
        Maximum dyadic depth setting the shared interval count.
    max_interval_prop : float, default=0.5
        Maximum interval length as a proportion of series length.
    representations : tuple of str, default=("base", "diff1", "periodogram")
        Series representations to extract intervals from.
    drop_constant : bool, default=True
        Drop zero-variance columns before the heads.
    n_estimators : int, default=200
        Number of trees in the ExtraTrees head.
    max_features : float, default=0.1
        Per-split feature sampling fraction for the ExtraTrees head.
    train_estimate : bool, default=False
        If True, bag the ExtraTrees head for an out-of-bag train estimate (the
        train estimate uses the ExtraTrees head only, not the full ensemble).
    class_weight : dict, "balanced", "balanced_subsample" or None, default=None
        Class weights for the heads.
    random_state : int, RandomState instance or None, default=None
        Seed for the intervals and heads.
    n_jobs : int, default=1
        Number of jobs for the ExtraTrees head.
    """

    def __init__(
        self,
        features="minimal",
        heads=("extratrees", "ridge"),
        min_interval_length=3,
        max_interval_depth=6,
        max_interval_prop=0.5,
        representations=("base", "diff1", "periodogram"),
        drop_constant=True,
        n_estimators=200,
        max_features=0.1,
        train_estimate=False,
        class_weight=None,
        random_state=None,
        n_jobs=1,
    ):
        self.heads = heads
        super().__init__(
            features=features,
            interval_scheme="random",
            min_interval_length=min_interval_length,
            max_interval_depth=max_interval_depth,
            max_interval_prop=max_interval_prop,
            banded=True,
            dilation=True,
            representations=representations,
            drop_constant=drop_constant,
            tree_type="extra",
            n_estimators=n_estimators,
            max_features=max_features,
            train_estimate=train_estimate,
            estimator=None,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def _fit_head(self, name, Xk, y):
        """Fit one classifier head on the (kept) feature matrix."""
        if name == "extratrees":
            est = _clone_estimator(self._default_estimator(), self.random_state)
            est.fit(Xk, y)
            return est
        if name == "ridge":
            # scaled Ridge; PULSAR's alpha grid
            pipe = make_pipeline(
                StandardScaler(),
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            )
            pipe.fit(Xk, y)
            return pipe
        raise ValueError(f"Unknown head '{name}', valid: {_VALID_HEADS}")

    @staticmethod
    def _head_proba(head, Xk):
        """Probabilities from a head (softmax of decision if no predict_proba)."""
        if hasattr(head, "predict_proba"):
            return head.predict_proba(Xk)
        decision = np.asarray(head.decision_function(Xk))
        if decision.ndim == 1:  # binary
            decision = np.column_stack((-decision, decision))
        decision = decision - decision.max(axis=1, keepdims=True)
        exp = np.exp(decision)
        return exp / exp.sum(axis=1, keepdims=True)

    def _fit(self, X, y):
        start = time.time()
        self._n_jobs = check_n_jobs(self.n_jobs)

        for name in self.heads:
            if name not in _VALID_HEADS:
                raise ValueError(f"Unknown head '{name}', valid: {_VALID_HEADS}")

        self._transformer = SharedIntervalTransform(
            features=self.features,
            interval_scheme="random",
            min_interval_length=self.min_interval_length,
            max_depth=self.max_interval_depth,
            max_interval_prop=self.max_interval_prop,
            banded=True,
            dilation=True,
            representations=self.representations,
            random_state=self.random_state,
        )
        X_t = self._transformer.fit_transform(X)

        if self.drop_constant:
            self._keep_cols_ = X_t.std(axis=0) > 0
            if not self._keep_cols_.any():
                self._keep_cols_ = np.ones(X_t.shape[1], dtype=bool)
        else:
            self._keep_cols_ = np.ones(X_t.shape[1], dtype=bool)
        self.n_features_used_ = int(self._keep_cols_.sum())
        Xk = X_t[:, self._keep_cols_]

        self._heads_ = {name: self._fit_head(name, Xk, y) for name in self.heads}
        # expose the ExtraTrees head as _estimator for the inherited OOB train
        # estimate and compatibility
        self._estimator = self._heads_.get("extratrees")

        self.fit_time_millis_ = int(round((time.time() - start) * 1000))
        return self

    def _predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]

    def _predict_proba(self, X) -> np.ndarray:
        Xk = self._transform_kept(X)
        probas = [self._head_proba(h, Xk) for h in self._heads_.values()]
        result = np.mean(probas, axis=0)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = result.sum(axis=1, keepdims=True)
        return np.divide(
            result,
            row_sums,
            out=np.full_like(result, 1 / self.n_classes_),
            where=row_sums != 0,
        )
