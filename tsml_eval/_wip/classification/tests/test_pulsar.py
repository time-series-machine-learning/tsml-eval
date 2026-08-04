"""Tests for the PULSAR classifier."""

import numpy as np

from tsml_eval._wip.classification import PULSARClassifier


def test_pulsar_supports_multivariate_collections():
    """PULSAR should build and reuse a feature block for every channel."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(12, 2, 16))
    y = np.repeat([0, 1], 6)
    X[y == 1, 1] += 2

    classifier = PULSARClassifier(
        representations=("original", "derivative"),
        interval_lengths=(3,),
        max_dilation=2,
        local_statistics=("mean", "stdev"),
        pooling_operators=("max", "mean"),
        hierarchical_depth=2,
        n_random_pooling_operators=1,
        feature_selection_percentage=40,
        classifiers=("ridge",),
        random_state=0,
    )
    probabilities = classifier.fit(X, y).predict_proba(X)

    assert classifier.get_tag("capability:multivariate") is True
    assert probabilities.shape == (12, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert set(classifier._states_by_channel_representation) == {
        (0, "original"),
        (0, "derivative"),
        (1, "original"),
        (1, "derivative"),
    }
    assert {item.channel for item in classifier.global_feature_metadata_} == {0, 1}
    assert {item.channel for item in classifier.candidate_feature_metadata_} == {
        0,
        1,
    }
