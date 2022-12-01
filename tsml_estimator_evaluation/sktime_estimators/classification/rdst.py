# -*- coding: utf-8 -*-
import numpy as np
from convst.classifiers import R_DST_Ensemble, R_DST_Ridge
from sktime.classification import BaseClassifier


class RDST(BaseClassifier):
    def __init__(
        self,
    ):
        super(RDST, self).__init__()

    def _fit(self, X, y):
        self.clf = R_DST_Ridge()
        self.clf.fit(X, y)

    def _predict(self, X) -> np.ndarray:
        return self.clf.predict(X)


class RDSTEnsemble(BaseClassifier):
    def __init__(
        self,
    ):
        super(RDSTEnsemble, self).__init__()

    def _fit(self, X, y):
        self.clf = R_DST_Ensemble()
        self.clf.fit(X, y)

    def _predict(self, X) -> np.ndarray:
        return self.clf.predict(X)
