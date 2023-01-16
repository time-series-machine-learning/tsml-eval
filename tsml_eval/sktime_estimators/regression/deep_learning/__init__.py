# -*- coding: utf-8 -*-
"""Deep learning based regressors."""
__all__ = [
    "ResNetRegressor",
    "InceptionTimeRegressor",
    "IndividualInceptionTimeRegressor",
    "FCNRegressor",
]

from tsml_eval.sktime_estimators.regression.deep_learning.fcn import FCNRegressor
from tsml_eval.sktime_estimators.regression.deep_learning.inception_time import (
    InceptionTimeRegressor,
    IndividualInceptionTimeRegressor,
)
from tsml_eval.sktime_estimators.regression.deep_learning.resnet import ResNetRegressor
