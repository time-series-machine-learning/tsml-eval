"""Tests that soft dependencies are handled correctly in modules."""

import re
from importlib import import_module

import pytest

from tsml_eval.testing.tests import ALL_TSML_EVAL_MODULES


def test_module_crawl():
    """Test that we are crawling modules correctly."""
    assert "tsml_eval.experiments" in ALL_TSML_EVAL_MODULES
    assert "tsml_eval.estimators" in ALL_TSML_EVAL_MODULES
    assert "tsml_eval.estimators.classification" in ALL_TSML_EVAL_MODULES


@pytest.mark.parametrize("module", ALL_TSML_EVAL_MODULES)
def test_module_soft_deps(module):
    """Test soft dependency imports in tsml-eval modules.

    Imports all modules and catch exceptions due to missing dependencies.
    """
    try:
        import_module(module)
    except ModuleNotFoundError as e:  # pragma: no cover
        dependency = "unknown"
        match = re.search(r"\'(.+?)\'", str(e))
        if match:
            dependency = match.group(1)

        raise ModuleNotFoundError(
            f"The module: {module} should not require any soft dependencies, "
            f"but tried importing: '{dependency}'. Make sure soft dependencies are "
            f"properly isolated."
        ) from e
