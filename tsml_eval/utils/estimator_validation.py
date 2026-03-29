"""Utilities for validating estimators."""

__all__ = [
    "is_sklearn_estimator",
    "is_sklearn_classifier",
    "is_sklearn_regressor",
    "is_sklearn_clusterer",
]

from aeon.base import BaseAeonEstimator
from aeon.utils.validation._dependencies import _check_soft_dependencies
from sklearn.base import BaseEstimator, is_classifier, is_clusterer, is_regressor


def is_sklearn_estimator(estimator):
    """Check if estimator is a scikit-learn estimator."""
    if _check_soft_dependencies("tsml", severity=None):
        from tsml.base import BaseTimeSeriesEstimator

        if isinstance(estimator, BaseTimeSeriesEstimator):
            return False

    return (
        isinstance(estimator, BaseEstimator)
        and not isinstance(estimator, BaseAeonEstimator)
        and not isinstance(estimator, BaseTimeSeriesEstimator)
    )


def is_sklearn_classifier(classifier):
    """Check if estimator is a scikit-learn classifier."""
    return is_sklearn_estimator(classifier) and is_classifier(classifier)


def is_sklearn_regressor(regressor):
    """Check if estimator is a scikit-learn regressor."""
    return is_sklearn_estimator(regressor) and is_regressor(regressor)


def is_sklearn_clusterer(clusterer):
    """Check if estimator is a scikit-learn clusterer."""
    return is_sklearn_estimator(clusterer) and is_clusterer(clusterer)
