# -*- coding: utf-8 -*-
"""Dummy time series regressors."""
__all__ = ["MeanPredictorRegressor", "MedianPredictorRegressor"]

from tsml_eval.estimators.regression.dummy.mean_predictor import MeanPredictorRegressor
from tsml_eval.estimators.regression.dummy.median_predictor import (
    MedianPredictorRegressor,
)
