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
__all__ = ["FixedWeightArsenal", "CVWeightArsenal", "EqualWeightArsenal"]

import numpy as np
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
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


def _binary_loo_accuracies(ridge, y):
    """Per-alpha leave-one-out accuracy, reconstructed from the stored predictions.

    ``RidgeClassifierCV`` already computes leave-one-out predictions for every alpha in
    closed form, and with a scorer set it stores them in ``cv_results_``. Only the
    reconstruction of labels from them is wrong for two classes: the stored column is a
    signed decision value, so the predicted class is its sign, where scikit-learn takes
    ``argmax`` over the single column and so always returns the first class. Redoing the
    reconstruction here recovers the correct accuracy at closed-form cost.
    """
    cv_results = getattr(ridge, "cv_results_", None)
    if cv_results is None or cv_results.ndim != 3 or cv_results.shape[1] != 1:
        return None
    positive = y == ridge.classes_[1]
    decisions = cv_results[:, 0, :]
    return np.array(
        [(np.sign(decisions[:, i]) > 0) == positive for i in range(decisions.shape[1])]
    ).mean(axis=1)


def _fit_ridge_binary_loo(X, y, class_weight):
    """Fit a binary member, scoring alphas by closed-form leave-one-out accuracy."""
    search = RidgeClassifierCV(
        alphas=ALPHAS,
        class_weight=class_weight,
        scoring="accuracy",
        store_cv_results=True,
    )
    search.fit(X, y)
    accuracies = _binary_loo_accuracies(search, y)
    if accuracies is None:
        return None

    best = int(np.argmax(accuracies))
    # Refitting on the chosen alpha is one ridge fit; explicit cross-validation would be
    # n_splits * n_alphas of them.
    ridge = RidgeClassifier(alpha=ALPHAS[best], class_weight=class_weight).fit(X, y)
    ridge.best_score_ = float(accuracies[best])
    ridge.alpha_ = ALPHAS[best]
    ridge.weighting_path_ = "loo-binary"
    return ridge


def _fit_ridge_binary_cv(X, y, class_weight, n_splits, path):
    """Fit a binary member by explicit stratified cross-validation."""
    ridge = RidgeClassifierCV(
        alphas=ALPHAS, class_weight=class_weight, scoring="accuracy", cv=n_splits
    )
    ridge.fit(X, y)
    ridge.weighting_path_ = path
    return ridge


def _fit_ridge_classifier(X, y, class_weight, binary_strategy="loo"):
    """Fit the member ridge so that ``best_score_`` is a usable accuracy estimate.

    Binary problems need their leave-one-out accuracy recovering, either in closed form
    or by explicit cross-validation. Multiclass problems keep the unmodified fast path,
    where scikit-learn's reconstruction is already correct.
    """
    n_classes = len(np.unique(y))
    n_splits = _n_splits_for(y)

    if n_classes == 2:
        if binary_strategy == "loo":
            try:
                ridge = _fit_ridge_binary_loo(X, y, class_weight)
            except (np.linalg.LinAlgError, TypeError):
                ridge = None
            if ridge is not None:
                return ridge
        if n_splits:
            return _fit_ridge_binary_cv(X, y, class_weight, n_splits, "cv-binary")
        # A singleton class leaves no way to estimate anything; the fast path, wrong
        # score and all, is all that remains, and the flag records that.

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
        return _fit_ridge_binary_cv(X, y, class_weight, n_splits, "cv-fallback")


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

    _binary_strategy = "loo"

    def _fit_ensemble_estimator(self, rocket, X, y, train_rng=None):
        rocket.fit(X)
        transformed_x = _transform_with(rocket, X, self.rocket_transform == "rocket")
        scaler = StandardScaler(with_mean=False)
        ridge = _fit_ridge_classifier(
            scaler.fit_transform(transformed_x),
            y,
            self.class_weight,
            binary_strategy=self._binary_strategy,
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
            scaler.fit_transform(Xt[subsample]),
            y[subsample],
            self.class_weight,
            binary_strategy=self._binary_strategy,
        )
        preds = ridge.predict(scaler.transform(Xt[oob]))
        return np.searchsorted(self.classes_, preds), ridge.best_score_, oob

    def _record_path(self, ridge):
        if not hasattr(self, "weighting_paths_"):
            self.weighting_paths_ = []
        self.weighting_paths_.append(getattr(ridge, "weighting_path_", "unknown"))


class CVWeightArsenal(FixedWeightArsenal):
    """Arsenal weighting binary members by explicit stratified cross-validation.

    A reference for ``FixedWeightArsenal``, which recovers the same quantity from the
    closed-form leave-one-out predictions. This variant refits the ridge for every fold
    and every alpha, so it is substantially slower, and is kept to confirm that the
    cheap reconstruction agrees with a direct estimate.
    """

    _binary_strategy = "cv"


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
