"""Focused tests for the experimental TDE per-dimension late fusion."""

import numpy as np

from aeon.classification.dictionary_based import IndividualTDE
from aeon.testing.data_generation import make_example_3d_numpy

from tsml_eval._wip.tde_dev._tde_dev3 import (
    IndividualTDE_Dev3,
    TDE_Dev3,
    normalised_dimension_similarity,
    normalised_histogram_intersection,
)


def _representation(dimensions):
    """Create the compact dimension-major representation used by Dev3 kernels."""
    key1_parts = []
    key2_parts = []
    count_parts = []
    offsets = []
    starts = []
    masses = []
    start = 0
    for cases in dimensions:
        starts.append(start)
        dim_offsets = [0]
        dim_masses = []
        for words, counts in cases:
            key1_parts.extend([0] * len(words))
            key2_parts.extend(words)
            count_parts.extend(counts)
            dim_offsets.append(dim_offsets[-1] + len(words))
            dim_masses.append(sum(counts))
        offsets.append(dim_offsets)
        masses.append(dim_masses)
        start += dim_offsets[-1]
    return (
        np.asarray(key1_parts, dtype=np.uint64),
        np.asarray(key2_parts, dtype=np.uint64),
        np.asarray(count_parts, dtype=np.uint32),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
        np.asarray(masses, dtype=np.float64),
    )


def test_normalised_histogram_intersection_hand_calculation():
    """Intersection is divided by the smaller non-empty bag mass."""
    first = _representation([[([1, 2], [2, 1])]])
    second = _representation([[([1, 3], [1, 2])]])

    similarity = normalised_histogram_intersection(
        first[0],
        first[1],
        first[2],
        0,
        2,
        3,
        second[0],
        second[1],
        second[2],
        0,
        2,
        3,
    )

    assert similarity == 1 / 3


def test_accuracy_weighted_dimension_similarity_hand_calculation():
    """The kernel applies the supplied normalised dimension weights exactly."""
    first = _representation(
        [
            [([1, 2], [2, 1])],
            [([5], [4])],
        ]
    )
    second = _representation(
        [
            [([1, 3], [1, 2])],
            [([5], [2])],
        ]
    )

    similarity = normalised_dimension_similarity(
        *first, 0, *second, 0, np.asarray([0.25, 0.75])
    )

    assert np.isclose(similarity, 0.25 * (1 / 3) + 0.75)


def test_normalisation_prevents_large_bag_mass_dominance():
    """A high-mass dimension cannot dominate only through its count scale."""
    query = _representation([[([1], [2])], [([9], [100])]])
    informative_match = _representation([[([1], [2])], [([9], [1])]])
    large_mass_match = _representation([[([2], [2])], [([9], [100])]])
    weights = np.asarray([0.5, 0.5])

    informative_similarity = normalised_dimension_similarity(
        *query, 0, *informative_match, 0, weights
    )
    large_mass_similarity = normalised_dimension_similarity(
        *query, 0, *large_mass_match, 0, weights
    )

    assert 2 + 1 < 0 + 100  # raw merged intersection favours bag mass
    assert informative_similarity > large_mass_similarity


def test_empty_dimension_bags_are_safe():
    """Empty dimensions contribute zero without division-by-zero or NaNs."""
    empty = _representation([[([], [])], [([2], [3])]])
    other = _representation([[([1], [4])], [([2], [3])]])

    similarity = normalised_dimension_similarity(
        *empty, 0, *other, 0, np.asarray([0.5, 0.5])
    )

    assert np.isfinite(similarity)
    assert similarity == 0.5


def test_default_merged_behaviour_matches_aeon():
    """Dev3 defaults retain the current aeon multivariate implementation."""
    X, y = make_example_3d_numpy(
        n_cases=14, n_channels=3, n_timepoints=30, random_state=3
    )
    params = {
        "window_size": 8,
        "word_length": 8,
        "bigrams": False,
        "dim_threshold": 0.0,
        "max_dims": 3,
        "random_state": 7,
    }
    baseline = IndividualTDE(**params).fit(X, y)
    dev3 = IndividualTDE_Dev3(**params).fit(X, y)

    assert dev3.multivariate_similarity == "merged"
    assert dev3.dimension_weighting == "uniform"
    assert dev3._dims == baseline._dims
    for actual, expected in zip(dev3._transformed_data, baseline._transformed_data):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(dev3.predict(X[:5]), baseline.predict(X[:5]))


def test_univariate_predictions_are_unchanged():
    """The experimental controls have no effect on univariate TDE."""
    X, y = make_example_3d_numpy(
        n_cases=14, n_channels=1, n_timepoints=30, random_state=4
    )
    params = {
        "window_size": 8,
        "word_length": 8,
        "bigrams": True,
        "random_state": 9,
    }
    baseline = IndividualTDE(**params).fit(X, y)
    dev3 = IndividualTDE_Dev3(
        **params,
        multivariate_similarity="normalised",
        dimension_weighting="accuracy",
    ).fit(X, y)

    for actual, expected in zip(dev3._transformed_data, baseline._transformed_data):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(dev3.predict(X[:5]), baseline.predict(X[:5]))


def test_multivariate_normalised_ensemble_fit_predict_and_proba():
    """The new similarity supports the complete ensemble estimator interface."""
    X, y = make_example_3d_numpy(
        n_cases=16, n_channels=3, n_timepoints=30, random_state=5
    )
    classifier = TDE_Dev3(
        n_parameter_samples=2,
        max_ensemble_size=2,
        randomly_selected_params=1,
        min_window=6,
        dim_threshold=0.0,
        max_dims=3,
        multivariate_similarity="normalised",
        dimension_weighting="accuracy",
        random_state=11,
    ).fit(X, y)

    predictions = classifier.predict(X[:4])
    probabilities = classifier.predict_proba(X[:4])

    assert predictions.shape == (4,)
    assert probabilities.shape == (4, classifier.n_classes_)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    for estimator in classifier.estimators_:
        assert np.isclose(estimator._dimension_weights_.sum(), 1)
        assert np.all(np.isfinite(estimator._dimension_weights_))
        score_sum = estimator._dimension_scores_.sum()
        expected = (
            estimator._dimension_scores_ / score_sum
            if score_sum > 0
            else np.full(len(estimator._dims), 1 / len(estimator._dims))
        )
        np.testing.assert_allclose(estimator._dimension_weights_, expected)


def test_normalised_mode_is_deterministic():
    """Fixed seeds give identical selected dimensions, weights and predictions."""
    X, y = make_example_3d_numpy(
        n_cases=16, n_channels=4, n_timepoints=30, random_state=6
    )
    params = {
        "window_size": 8,
        "word_length": 8,
        "bigrams": False,
        "dim_threshold": 0.0,
        "max_dims": 3,
        "multivariate_similarity": "normalised",
        "dimension_weighting": "accuracy",
        "random_state": 12,
    }
    first = IndividualTDE_Dev3(**params).fit(X, y)
    second = IndividualTDE_Dev3(**params).fit(X, y)

    assert first._dims == second._dims
    np.testing.assert_array_equal(
        first._dimension_scores_, second._dimension_scores_
    )
    np.testing.assert_array_equal(
        first._dimension_weights_, second._dimension_weights_
    )
    np.testing.assert_array_equal(first.predict(X[:6]), second.predict(X[:6]))
