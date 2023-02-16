# -*- coding: utf-8 -*-
import numpy as np
from mrsqm import MrSQMClassifier
from sktime.classification import BaseClassifier


class MrSQM(BaseClassifier):
    def __init__(
        self,
        random_state=None,
    ):
        self.random_state = random_state
        super(MrSQM, self).__init__()

    def _fit(self, X, y):
        self.clf = MrSQMClassifier(random_state=self.random_state)
        self.clf.fit(X, y)

    def _predict(self, X) -> np.ndarray:
        return self.clf.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)
