"""Tests for data transforms in experiments."""

import pytest

from tsml_eval.experiments import get_data_transform_by_name, _get_data_transform
from tsml_eval.testing.testing_utils import _check_set_method, _check_set_method_results


def test_get_data_transform_by_name():
    """Test get_data_transform_by_name method."""
    transform_lists = [
        _get_data_transform.transformers
    ]

    transform_dict = {}
    all_transform_names = []

    for transform_list in transform_lists:
        _check_set_method(
            get_data_transform_by_name,
            transform_list,
            transform_dict,
            all_transform_names,
        )

    _check_set_method_results(
        transform_dict, estimator_name="Transformers", method_name="get_data_transform_by_name"
    )


def test_get_data_transform_by_name_multiple_output():
    assert False

def test_get_data_transform_by_name_invalid():
    """Test get_data_transform_by_name method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN TRANSFORMER"):
        get_data_transform_by_name("invalid")