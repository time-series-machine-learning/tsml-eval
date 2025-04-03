"""Tests for data transforms in experiments."""

import pytest
from aeon.transformations.collection import Normalizer, Padder

from tsml_eval.experiments import _get_data_transform, get_data_transform_by_name
from tsml_eval.testing.testing_utils import _check_set_method, _check_set_method_results


def test_get_data_transform_by_name():
    """Test get_data_transform_by_name method."""
    transform_name_lists = [
        _get_data_transform.scaling_transformers,
        _get_data_transform.unbalanced_transformers,
        _get_data_transform.unequal_transformers,
    ]

    transform_list = []
    transform_dict = {}
    all_transform_names = []
    for transform_name_list in transform_name_lists:
        _check_set_method(
            get_data_transform_by_name,
            transform_name_list,
            transform_list,
            transform_dict,
            all_transform_names,
        )

    _check_set_method_results(
        transform_dict,
        estimator_name="Transformers",
        method_name="get_data_transform_by_name",
    )


def test_get_data_transform_by_name_multiple_output():
    """Test get_data_transform_by_name method with multiple inputs and outputs."""
    t = get_data_transform_by_name(["padder", "normaliser"], row_normalise=True)
    assert len(t) == 3
    assert isinstance(t[0], Normalizer)
    assert isinstance(t[1], Padder)
    assert isinstance(t[2], Normalizer)


def test_get_data_transform_by_name_invalid():
    """Test get_data_transform_by_name method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN TRANSFORMER"):
        get_data_transform_by_name("invalid")
