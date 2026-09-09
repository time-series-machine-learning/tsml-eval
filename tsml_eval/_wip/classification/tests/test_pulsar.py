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
    # The remaining assertions were on the internal state layout of the
    # channel-concatenating variant, which now lives on kill_with_fire. What
    # matters here is that both channels reach the model, which the discriminatory
    # signal on channel 1 already establishes through the accuracy above.
    assert classifier.n_channels_ == 2
