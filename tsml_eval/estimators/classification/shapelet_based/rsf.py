# -*- coding: utf-8 -*-
import numpy as np
from aeon.classification import BaseClassifier
from wildboar.ensemble import ShapeletForestClassifier


class RandomShapeletForest(BaseClassifier):
    def __init__(
        self,
        random_state=None,
    ):
        self.random_state = random_state
        super(RandomShapeletForest, self).__init__()

    def _fit(self, X, y):
        if X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        self.clf = ShapeletForestClassifier(random_state=self.random_state, n_jobs=1)
        self.clf.fit(X, y)

    def _predict(self, X) -> np.ndarray:
        if X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        return self.clf.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        if X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        return self.clf.predict_proba(X)
