# -*- coding: utf-8 -*-
"""Deep learning based classifiers."""
__all__ = [
    "CNNClassifier",
    "InceptionTimeClassifier",
]

from tsml_eval.sktime_estimators.classification.deep_learning.cnn import (
    CNNClassifier,
    InceptionTimeClassifier,
)
