# -*- coding: utf-8 -*-
import numpy as np
from convst.classifiers import R_DST_Ensemble, R_DST_Ridge
from sktime.classification import BaseClassifier


class RDST(BaseClassifier):
    def __init__(
        self,
        random_state=None,
    ):
        self.random_state = random_state
        super(RDST, self).__init__()

    def _fit(self, X, y):
        self.clf = R_DST_Ridge(random_state=self.random_state)
        self.clf.fit(X, y)

    def _predict(self, X) -> np.ndarray:
        return self.clf.predict(X)


class RDSTEnsemble(BaseClassifier):
    def __init__(
        self,
        random_state=None,
    ):
        self.random_state = random_state
        super(RDSTEnsemble, self).__init__()

    def _fit(self, X, y):
        self.clf = R_DST_Ensemble(random_state=self.random_state)
        self.clf.fit(X, y)

    def _predict(self, X) -> np.ndarray:
        return self.clf.predict(X)
