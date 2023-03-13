# -*- coding: utf-8 -*-
"""Test estimators implemented in tsml-eval."""

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tsml.utils.testing import parametrize_with_checks

from tsml_eval.estimators.classification.sklearn_classifier import (
    SklearnToTsmlClassifier,
)
from tsml_eval.estimators.clustering.sklearn_clusterer import SklearnToTsmlClusterer
from tsml_eval.estimators.regression.sklearn_regressor import SklearnToTsmlRegressor


@parametrize_with_checks(
    [
        SklearnToTsmlClassifier(RandomForestClassifier(n_estimators=5)),
        SklearnToTsmlRegressor(RandomForestRegressor(n_estimators=5)),
        SklearnToTsmlClusterer(KMeans(n_clusters=2, max_iter=5)),
    ]
)
def test_tsml_wrapper_estimator(estimator, check):
    """Test that tsml estimators adhere to tsml conventions."""
    check(estimator)
