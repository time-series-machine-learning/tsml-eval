"""QUANT classifier with bagging and DrCIF-style OOB train estimates.

Clone of aeon's QUANTClassifier (aeon 1.4.0) where the default extra trees
ensemble is built with bootstrap sampling. Each tree is trained on a bootstrap
sample of cases (bagging, as DrCIF's train estimate does per tree) and train
probability estimates come from the out-of-bag decision function, giving the
classifier the capability:train_estimate tag so fit_predict does not fall back
to 10-fold cross-validation.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["BaggedQUANT"]

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.interval_based import QUANTTransformer


class BaggedQUANT(BaseClassifier):
    """QUANT interval classifier with a bagged ensemble and OOB train estimates.

    Identical to aeon's QUANTClassifier except the default ExtraTreesClassifier
    is built with ``bootstrap=True`` and ``oob_score=True``: each tree sees a
    bootstrap sample of the training cases and train estimates are taken from
    the out-of-bag decision function, mirroring how DrCIF produces its train
    estimates. Cases never out-of-bag get uniform probabilities, as in DrCIF.

    Parameters
    ----------
    interval_depth : int, default=6
        The depth to stop extracting intervals at.
    quantile_divisor : int, default=4
        The divisor to find the number of quantiles to extract from intervals.
    estimator : sklearn estimator or None, default=None
        The estimator to use for classification. If None, an
        ExtraTreesClassifier with 200 trees and bootstrap sampling is used.
        A user-supplied estimator must set bootstrap=True and oob_score=True
        for train estimates to be available.
    class_weight : dict, "balanced", "balanced_subsample" or None, default=None
        Only applies to the default ExtraTreesClassifier.
    random_state : int, RandomState instance or None, default=None
        Seed or RandomState for reproducibility.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "algorithm_type": "interval",
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        interval_depth=6,
        quantile_divisor=4,
        estimator=None,
        class_weight=None,
        random_state=None,
    ):
        self.interval_depth = interval_depth
        self.quantile_divisor = quantile_divisor
        self.estimator = estimator
        self.class_weight = class_weight
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        self._transformer = QUANTTransformer(
            interval_depth=self.interval_depth,
            quantile_divisor=self.quantile_divisor,
        )

        self._estimator = _clone_estimator(
            (
                ExtraTreesClassifier(
                    n_estimators=200,
                    max_features=0.1,
                    criterion="entropy",
                    class_weight=self.class_weight,
                    bootstrap=True,
                    oob_score=True,
                    random_state=self.random_state,
                )
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        return self

    def _fit_predict(self, X, y) -> np.ndarray:
        probas = self._fit_predict_proba(X, y)
        return np.array(
            [self.classes_[int(np.argmax(prob))] for prob in probas]
        )

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
        return {"interval_depth": 2, "quantile_divisor": 8}
