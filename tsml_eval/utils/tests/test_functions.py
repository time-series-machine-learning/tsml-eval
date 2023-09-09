"""Test utility functions."""

from tsml_eval.utils.functions import pair_list_to_dict


def test_pair_list_to_dict():
    """Test pair_list_to_dict function."""
    assert pair_list_to_dict([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}
    assert pair_list_to_dict(None) == {}
