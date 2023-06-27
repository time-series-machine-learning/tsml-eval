# -*- coding: utf-8 -*-
"""Deep learning based classifiers."""

__all__ = [
    "CNNClassifier",
    "InceptionTimeClassifier",
    "IndividualInceptionTimeClassifier",
    "ResNetClassifier",
]

from tsml_eval.estimators.classification.deep_learning.cnn import CNNClassifier
from tsml_eval.estimators.classification.deep_learning.inception_time import (
    InceptionTimeClassifier,
    IndividualInceptionTimeClassifier,
)
from tsml_eval.estimators.classification.deep_learning.resnet import ResNetClassifier
