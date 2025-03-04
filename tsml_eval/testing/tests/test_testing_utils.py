"""Test testing utilities."""

import pytest

from tsml_eval.testing.testing_utils import (
    _check_set_method,
    _check_set_method_results,
    suppress_output,
)


def test_check_set_method_fail():
    """Test _check_set_method_ failure state."""
    with pytest.raises(ModuleNotFoundError):
        _check_set_method(
            _test_set_method_fail,
            ["a", "b", "c"],
            [],
            {},
            [],
        )


def _test_set_method_fail(estimator_alias):
    raise ModuleNotFoundError(
        "Just imagine you imported a module which does not exist when "
        f"importing {estimator_alias}."
    )


def test_check_set_method_results_fail():
    """Test _check_set_method_results failure state."""
    with pytest.raises(ValueError, match=r"missing entries: \['b'\]"):
        _check_set_method_results({"a": True, "b": False, "c": True})


@suppress_output(suppress_stdout=False, suppress_stderr=False)
def test_suppress_output_false():
    """Test suppress_output method with False inputs."""
    pass
