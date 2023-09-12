"""Test experiment utilities."""

import pytest

from tsml_eval.utils.experiments import _results_present


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
