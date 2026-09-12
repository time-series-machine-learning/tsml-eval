"""Arsenal with a member weighting that is correct for binary problems.

Arsenal weights each ensemble member by ``RidgeClassifierCV.best_score_``, fitted with
``scoring="accuracy"``. For a binary problem that attribute is exactly 1.0 whatever the
data, so every member receives the same weight and the ensemble is unweighted. The cause
is upstream: with ``cv=None`` the efficient generalised cross-validation path
reconstructs predicted labels with ``argmax(axis=1)`` over the single column that
``LabelBinarizer`` produces for two classes, which always returns the first class. See
https://github.com/scikit-learn/scikit-learn/issues/34942 (open at the time of writing).

Passing an explicit ``cv`` avoids that path entirely and returns a correct mean
cross-validated accuracy, so the fix here is to route binary problems through explicit
stratified cross-validation and leave multiclass problems on the fast path, where
``best_score_`` is already correct. Multiclass behaviour is therefore unchanged.

This is deliberately a subclass in the WIP area rather than a change to ``aeon``: it
allows the fix to be evaluated against the released Arsenal on the archive before being
proposed upstream.

``EqualWeightArsenal`` is the control for the same question: it discards the member
weights entirely and takes a straight vote. On a binary problem the released Arsenal is
already an unweighted vote, since every weight is 1.0, so the two agree there by
construction; they differ only when there are more than two classes. Comparing all three
separates "are the weights correct" from "do the weights earn their cost at all".
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["FixedWeightArsenal", "EqualWeightArsenal"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.classification.convolution_based import Arsenal

# The member fitting this subclass overrides only exists in the reworked Arsenal. On an
# older aeon the whole structure differs (different signature, weights computed
# elsewhere) and silently inheriting it would produce an estimator that looks fixed and
# is not, so the import failure is captured and reported at construction instead.
try:
    from aeon.classification.convolution_based._arsenal import (
        _fit_ridge_classifier as _aeon_fit_ridge,
        _get_oob_indices,
        _transform_with,
    )

    _AEON_SUPPORTED = True
except ImportError:  # pragma: no cover - depends on the installed aeon
    _AEON_SUPPORTED = False
    _aeon_fit_ridge = _get_oob_indices = _transform_with = None

ALPHAS = np.logspace(-3, 3, 10)
MAX_SPLITS = 5


def _n_splits_for(y):
    """Largest usable number of stratified folds, or 0 if CV is not possible."""
    _, counts = np.unique(y, return_counts=True)
    n_splits = min(MAX_SPLITS, int(counts.min()))
    return n_splits if n_splits >= 2 else 0


def _fit_ridge_classifier(X, y, class_weight):
    """Fit the member ridge so that ``best_score_`` is a usable accuracy estimate.

    Binary problems always use explicit cross-validation, because the generalised
    cross-validation path reports 1.0 for them regardless of the data. Multiclass
    problems keep the fast path, falling back when its SVD does not converge.
    """
    n_classes = len(np.unique(y))
    n_splits = _n_splits_for(y)

    # Binary is only routed away from the fast path when folds are actually possible;
    # with a singleton class no cross-validation exists and the fast path, wrong score
    # and all, is still the only option. The flag records which happened.
    if n_classes == 2 and n_splits:
        ridge = RidgeClassifierCV(
            alphas=ALPHAS,
            class_weight=class_weight,
            scoring="accuracy",
            cv=n_splits,
        )
        ridge.fit(X, y)
        ridge.weighting_path_ = "cv-binary"
        return ridge

    ridge = RidgeClassifierCV(
        alphas=ALPHAS, class_weight=class_weight, scoring="accuracy"
    )
    try:
        ridge.fit(X, y)
        ridge.weighting_path_ = "gcv" if n_classes > 2 else "gcv-degenerate"
        return ridge
    except np.linalg.LinAlgError:
        if not n_splits:
            raise
        ridge = RidgeClassifierCV(
            alphas=ALPHAS,
            class_weight=class_weight,
            scoring="accuracy",
            cv=n_splits,
        )
        ridge.fit(X, y)
        ridge.weighting_path_ = "cv-fallback"
        return ridge


class FixedWeightArsenal(Arsenal):
    """Arsenal whose member weights are correct for binary problems.

    Identical to ``aeon``'s ``Arsenal`` except for how each member's weight is
    estimated. On problems with more than two classes the two are equivalent.

    Attributes
    ----------
    weighting_paths_ : list of str
        Which estimation path each member used, one of ``"cv-binary"``, ``"gcv"``,
        ``"cv-fallback"`` or ``"gcv-degenerate"``. Recorded so an experiment can verify
        that the intended path was taken.
    """

    def _fit(self, X, y):
        if not _AEON_SUPPORTED:
            import aeon

            raise RuntimeError(
                "FixedWeightArsenal requires an aeon whose Arsenal fits its members "
                "through _fit_ridge_classifier; the installed aeon is "
                f"{aeon.__version__}, whose Arsenal has a different structure and "
                "weights members by a negative squared error rather than an accuracy. "
                "Install the aeon that this fix targets before running."
            )
        return super()._fit(X, y)

    def _fit_ensemble_estimator(self, rocket, X, y, train_rng=None):
        rocket.fit(X)
        transformed_x = _transform_with(rocket, X, self.rocket_transform == "rocket")
        scaler = StandardScaler(with_mean=False)
        ridge = _fit_ridge_classifier(
            scaler.fit_transform(transformed_x), y, self.class_weight
        )
        self._record_path(ridge)
        pipeline = make_pipeline(rocket, scaler, ridge)

        train_estimate = (
            self._train_probas_for_estimator(transformed_x, y, train_rng)
            if train_rng is not None
            else None
        )
        return pipeline, ridge.best_score_, train_estimate

    def _train_probas_for_estimator(self, Xt, y, rng):
        subsample = rng.choice(self.n_cases_, size=self.n_cases_)
        oob = _get_oob_indices(subsample, self.n_cases_)

        if oob.size == 0:
            return np.empty(0, dtype=np.intp), 0.0, oob

        scaler = StandardScaler(with_mean=False)
        ridge = _fit_ridge_classifier(
            scaler.fit_transform(Xt[subsample]), y[subsample], self.class_weight
        )
        preds = ridge.predict(scaler.transform(Xt[oob]))
        return np.searchsorted(self.classes_, preds), ridge.best_score_, oob

    def _record_path(self, ridge):
        if not hasattr(self, "weighting_paths_"):
            self.weighting_paths_ = []
        self.weighting_paths_.append(getattr(ridge, "weighting_path_", "unknown"))


class EqualWeightArsenal(FixedWeightArsenal):
    """Arsenal that ignores member quality and takes a straight vote.

    Every member contributes with weight one, so the ensemble probability is the plain
    mean of the members' one-hot predictions. This is the control for whether CAWPE
    weighting inside Arsenal earns its cost at all.

    On a binary problem the released ``Arsenal`` already behaves this way, because the
    upstream defect gives every member a weight of exactly one, so the two agree by
    construction. They differ only when there are more than two classes.

    Members are still fitted through the fast generalised cross-validation path, since
    the score it returns is discarded: this variant is therefore no more expensive than
    the released Arsenal, and cheaper than ``FixedWeightArsenal``.
    """

    def _fit_ensemble_estimator(self, rocket, X, y, train_rng=None):
        rocket.fit(X)
        transformed_x = _transform_with(rocket, X, self.rocket_transform == "rocket")
        scaler = StandardScaler(with_mean=False)
        # The score is never read, so the cheap path is always the right one here.
        ridge = _aeon_fit_ridge(scaler.fit_transform(transformed_x), y, self.class_weight)
        self._record_path_name("equal")
        pipeline = make_pipeline(rocket, scaler, ridge)

        train_estimate = (
            self._train_probas_for_estimator(transformed_x, y, train_rng)
            if train_rng is not None
            else None
        )
        return pipeline, 1.0, train_estimate

    def _train_probas_for_estimator(self, Xt, y, rng):
        subsample = rng.choice(self.n_cases_, size=self.n_cases_)
        oob = _get_oob_indices(subsample, self.n_cases_)

        if oob.size == 0:
            return np.empty(0, dtype=np.intp), 0.0, oob

        scaler = StandardScaler(with_mean=False)
        ridge = _aeon_fit_ridge(
            scaler.fit_transform(Xt[subsample]), y[subsample], self.class_weight
        )
        preds = ridge.predict(scaler.transform(Xt[oob]))
        return np.searchsorted(self.classes_, preds), 1.0, oob

    def _record_path_name(self, name):
        if not hasattr(self, "weighting_paths_"):
            self.weighting_paths_ = []
        self.weighting_paths_.append(name)
