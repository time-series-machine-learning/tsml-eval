"""Tests for publication experiments estimator selection."""

import pytest

from tsml_eval.publications.y2023.distance_based_clustering import (
    _set_distance_clusterer,
    distance_based_clusterers,
)
from tsml_eval.testing.testing_utils import _check_set_method, _check_set_method_results


def test_set_distance_clusterer():
    """Test set_distance_clusterer method."""
    clusterer_list = []
    clusterer_dict = {}
    all_clusterer_names = []
    _check_set_method(
        _set_distance_clusterer,
        distance_based_clusterers,
        clusterer_list,
        clusterer_dict,
        all_clusterer_names,
    )

    _check_set_method_results(
        clusterer_dict,
        estimator_name="Clusterers",
        method_name="_set_distance_clusterer",
    )


def test_set_distance_clusterer_invalid():
    """Test set_distance_clusterer method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN CLUSTERER"):
        _set_distance_clusterer("invalid")
