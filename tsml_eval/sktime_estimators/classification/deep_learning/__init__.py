# -*- coding: utf-8 -*-
"""Deep learning based classifiers."""

__all__ = [
    "InceptionTimeClassifier",
    "IndividualInceptionTimeClassifier",
]

from tsml_eval.sktime_estimators.classification.deep_learning.inception_time import (
    InceptionTimeClassifier,
    IndividualInceptionTimeClassifier,
)
