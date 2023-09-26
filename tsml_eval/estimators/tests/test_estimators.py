"""Test estimators implemented in tsml-eval."""

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tsml.utils.testing import parametrize_with_checks

from tsml_eval.estimators import (
    SklearnToTsmlClassifier,
    SklearnToTsmlClusterer,
    SklearnToTsmlRegressor,
)


@parametrize_with_checks(
    [
        SklearnToTsmlClassifier(classifier=RandomForestClassifier(n_estimators=5)),
        SklearnToTsmlRegressor(regressor=RandomForestRegressor(n_estimators=5)),
        SklearnToTsmlClusterer(clusterer=KMeans(n_clusters=2, max_iter=5)),
    ]
)
def test_tsml_wrapper_estimator(estimator, check):
    """Test that tsml wrapper estimators adhere to tsml conventions."""
    check(estimator)
