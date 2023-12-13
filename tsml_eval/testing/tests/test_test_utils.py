"""Test testing utilities."""

import pytest

from tsml_eval.testing.test_utils import _check_set_method_results


def test_check_set_method_results_fail():
    """Test _check_set_method_results failure state."""
    with pytest.raises(ValueError, match=r"missing entries: \['b'\]"):
        _check_set_method_results({"a": True, "b": False, "c": True})
