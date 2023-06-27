# -*- coding: utf-8 -*-
import numpy as np
from aeon.classification import BaseClassifier


class MrSQM(BaseClassifier):
    _tags = {
        "X_inner_mtype": "nested_univ",  # input in nested dataframe
    }

    def __init__(
        self,
        random_state=None,
    ):
        self.random_state = random_state
        super(MrSQM, self).__init__()

    def _fit(self, X, y):
        from mrsqm import MrSQMClassifier

        self.clf = MrSQMClassifier(random_state=self.random_state, nsax=0, nsfa=5)
        self.clf.fit(X, y)

    def _predict(self, X) -> np.ndarray:
        return self.clf.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)
