# -*- coding: utf-8 -*-
"""Deep learning based regressors."""
__all__ = [
    "CNNRegressor",
    "FCNRegressor",
    "ResNetRegressor",
    "InceptionTimeRegressor",
    "IndividualInceptionTimeRegressor",
]

from tsml_eval.estimators.regression.deep_learning.cnn import CNNRegressor
from tsml_eval.estimators.regression.deep_learning.fcn import FCNRegressor
from tsml_eval.estimators.regression.deep_learning.inception_time import (
    InceptionTimeRegressor,
    IndividualInceptionTimeRegressor,
)
from tsml_eval.estimators.regression.deep_learning.resnet import ResNetRegressor
