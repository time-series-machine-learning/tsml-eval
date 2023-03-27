# -*- coding: utf-8 -*-
import numpy as np
from aeon.classification import BaseClassifier


class RDST(BaseClassifier):
    def __init__(
        self,
        n_shapelets=10000,
        random_state=None,
    ):
        self.n_shapelets = n_shapelets
        self.random_state = random_state
        super(RDST, self).__init__()

    def _fit(self, X, y):
        from convst.classifiers import R_DST_Ridge

        self.clf = R_DST_Ridge(
            n_shapelets=self.n_shapelets, random_state=self.random_state
        )
        self.clf.fit(X, y)

    def _predict(self, X) -> np.ndarray:
        return self.clf.predict(X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {
            "n_shapelets": 20,
        }


class RDSTEnsemble(BaseClassifier):
    def __init__(
        self,
        n_shapelets=10000,
        random_state=None,
    ):
        self.n_shapelets = n_shapelets
        self.random_state = random_state
        super(RDSTEnsemble, self).__init__()

    def _fit(self, X, y):
        from convst.classifiers import R_DST_Ensemble

        self.clf = R_DST_Ensemble(
            n_shapelets_per_estimator=self.n_shapelets, random_state=self.random_state
        )
        self.clf.fit(X, y)

    def _predict(self, X) -> np.ndarray:
        return self.clf.predict(X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {
            "n_shapelets": 20,
        }
