"""Classifier using aeon's Bag-of-Receptive-Fields transform."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["BORFClassifier"]

from sklearn.linear_model import RidgeClassifierCV

from aeon.classification.base import BaseClassifier


class BORFClassifier(BaseClassifier):
    """Bag-of-Receptive-Fields classifier with the paper's ridge head.

    aeon exposes BORF as a multivariate collection transformer rather than a
    classifier. The original implementation composes BORF with
    ``RidgeClassifierCV``; this wrapper supplies that composition through the
    aeon classifier interface required by tsml-eval.

    ``RidgeClassifierCV`` has no ``predict_proba`` method. Consequently this
    class deliberately inherits :class:`BaseClassifier`'s standard probability
    fallback: the predicted class receives probability one and all other
    classes receive zero.

    Parameters
    ----------
    n_jobs : int, default=1
        Number of BORF configurations evaluated in parallel.
    n_jobs_numba : int, default=1
        Number of Numba threads used inside each BORF configuration.
    complexity : {"quadratic", "linear"}, default="quadratic"
        BORF configuration-search complexity. ``"quadratic"`` is the aeon
        default and the accuracy-oriented configuration.
    random_state : int or None, default=None
        Accepted for the common classifier factory interface. BORF and the
        ridge head are deterministic and do not use it.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:multithreading": True,
        "algorithm_type": "dictionary",
        "python_dependencies": "sparse",
        "cant_pickle": True,
        "non_deterministic": True,
    }

    def __init__(
        self,
        n_jobs=1,
        n_jobs_numba=1,
        complexity="quadratic",
        random_state=None,
    ):
        self.n_jobs = n_jobs
        self.n_jobs_numba = n_jobs_numba
        self.complexity = complexity
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        # Keep the BORF import lazy so unrelated classifiers remain usable with
        # older aeon installations. Only an explicit BORF run requires it.
        from aeon.transformations.collection.dictionary_based import BORF

        self.transformer_ = BORF(
            n_jobs=self.n_jobs,
            n_jobs_numba=self.n_jobs_numba,
            complexity=self.complexity,
        )
        Xt = self.transformer_.fit_transform(X, y)
        self.classifier_ = RidgeClassifierCV()
        self.classifier_.fit(Xt, y)
        return self

    def _predict(self, X):
        Xt = self.transformer_.transform(X)
        return self.classifier_.predict(Xt)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"n_jobs": 1, "n_jobs_numba": 1, "complexity": "linear"}
