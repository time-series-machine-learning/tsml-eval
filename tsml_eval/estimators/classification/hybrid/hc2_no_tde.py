"""HIVE-COTE 2 without TDE, fitted in memory."""

__all__ = ["HIVECOTEV2NoTDE"]

import numpy as np
from aeon.classification.base import BaseClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state


class HIVECOTEV2NoTDE(BaseClassifier):
    """HIVE-COTE 2 with Arsenal, DrCIF and STC, and an optional class-prior correction.

    The in-memory counterpart of ``HC2-noTDE-500-pc``, which is built from component
    results files by ``build_hivecote_from_results``. As in aeon's ``HIVECOTEV2``,
    each component is weighted by the accuracy of its train estimate (from
    ``fit_predict``) raised to ``alpha``, and the weighted probabilities are summed
    and normalised. With ``prior_correction`` (beta), the combined posterior is then
    divided by the train class prior raised to beta and renormalised.

    Parameters
    ----------
    arsenal_params : dict or None, default=None
        Parameters for the Arsenal component.
    drcif_params : dict or None, default=None
        Parameters for the DrCIF component. None gives ``{"n_estimators": 500}``.
    stc_params : dict or None, default=None
        Parameters for the ShapeletTransformClassifier component.
    alpha : float, default=4
        Exponent of the train-estimate accuracy in each component's weight.
    prior_correction : float, default=0.5
        Beta for dividing by the train class prior; 0 leaves the CAWPE rule unchanged.
    random_state : int, RandomState or None, default=None
        Seed for the components and for breaking ties between classes.
    n_jobs : int, default=1
        Threads for the components.

    Attributes
    ----------
    weights_ : list of float
        Component weights, in the order Arsenal, DrCIF, STC.
    prior_ : np.ndarray
        Train class prior, in ``classes_`` order.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "hybrid",
    }

    def __init__(
        self,
        arsenal_params=None,
        drcif_params=None,
        stc_params=None,
        alpha=4,
        prior_correction=0.5,
        random_state=None,
        n_jobs=1,
    ):
        self.arsenal_params = arsenal_params
        self.drcif_params = drcif_params
        self.stc_params = stc_params
        self.alpha = alpha
        self.prior_correction = prior_correction
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__()

    def _components(self):
        from aeon.classification.convolution_based import Arsenal
        from aeon.classification.interval_based import DrCIFClassifier
        from aeon.classification.shapelet_based import ShapeletTransformClassifier

        drcif = (
            {"n_estimators": 500} if self.drcif_params is None else self.drcif_params
        )
        shared = {"random_state": self.random_state, "n_jobs": self.n_jobs}
        return [
            Arsenal(**{**shared, **(self.arsenal_params or {})}),
            DrCIFClassifier(**{**shared, **drcif}),
            ShapeletTransformClassifier(**{**shared, **(self.stc_params or {})}),
        ]

    def _fit(self, X, y):
        self.estimators_, self.weights_ = [], []
        for estimator in self._components():
            # fit_predict fits the component and returns its train estimate
            train_preds = estimator.fit_predict(X, y)
            self.weights_.append(accuracy_score(y, train_preds) ** self.alpha)
            self.estimators_.append(estimator)
        counts = np.array([np.sum(y == c) for c in self.classes_])
        self.prior_ = counts / counts.sum()
        return self

    def _predict_proba(self, X):
        dists = np.zeros((len(X), self.n_classes_))
        for weight, estimator in zip(self.weights_, self.estimators_):
            dists += weight * estimator.predict_proba(X)
        totals = dists.sum(axis=1, keepdims=True)
        dists = np.divide(
            dists,
            totals,
            out=np.full_like(dists, 1 / self.n_classes_),
            where=totals > 0,
        )
        if self.prior_correction:
            dists = dists / self.prior_**self.prior_correction
            dists = dists / dists.sum(axis=1, keepdims=True)
        return dists

    def _predict(self, X):
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(p == p.max())))]
                for p in self._predict_proba(X)
            ]
        )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        from aeon.classification.sklearn import RotationForestClassifier

        return {
            "arsenal_params": {"n_kernels": 20, "n_estimators": 2},
            "drcif_params": {
                "n_estimators": 2,
                "n_intervals": 2,
                "att_subsample_size": 2,
            },
            "stc_params": {
                "estimator": RotationForestClassifier(n_estimators=2),
                "n_shapelet_samples": 20,
                "max_shapelets": 5,
                "batch_size": 10,
            },
        }
