# -*- coding: utf-8 -*-
"""Utilities for validating estimators."""

from aeon.base import BaseEstimator as AeonBaseEstimator
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import is_classifier, is_regressor
from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import is_clusterer


def is_sklearn_estimator(estimator):
    """Check if estimator is a scikit-learn estimator."""
    return (
        isinstance(estimator, SklearnBaseEstimator)
        and not isinstance(estimator, AeonBaseEstimator)
        and not isinstance(estimator, BaseTimeSeriesEstimator)
    )


def is_sklearn_classifier(classifier):
    """Check if estimator is a scikit-learn classifier."""
    return is_sklearn_estimator(classifier) and is_classifier(classifier)


def is_sklearn_regressor(regressor):
    """Check if estimator is a scikit-learn regressor."""
    return is_sklearn_estimator(regressor) and is_regressor(regressor)


def is_sklearn_clusterer(clusterer):
    """Check if estimator is a scikit-learn classifier."""
    return is_sklearn_estimator(clusterer) and is_clusterer(clusterer)
