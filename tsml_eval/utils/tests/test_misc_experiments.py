"""Test experiment utilities."""

import pytest

from tsml_eval.utils.experiments import _results_present, timing_benchmark


@pytest.mark.parametrize("split", ["BOTH", "TRAIN", "TEST", None, "invalid"])
def test_results_present_split_inputs(split):
    """Test _results_present function with valid and invalid split inputs."""
    if split == "invalid":
        with pytest.raises(ValueError, match="Unknown split value"):
            _results_present(
                "test_output",
                "test",
                "test",
                split=split,
            )
    else:
        assert not _results_present(
            "test_output",
            "test",
            "test",
            split=split,
        )


def test_timing_benchmark_invalid_input():
    """Test timing_benchmark function with invalid input."""
    with pytest.raises(ValueError):
        timing_benchmark(random_state="invalid")
