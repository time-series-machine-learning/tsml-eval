"""Run aeon estimator checks on all public aeon estimators."""

import importlib
import inspect
import pkgutil

from aeon.base import BaseAeonEstimator
from aeon.testing.estimator_checking import parametrize_with_checks

import tsml_eval.estimators as estimators_pkg

_SKIP_ESTIMATORS = ["FromFileHIVECOTE"]


def _get_estimators():
    estimators = []

    for _, module_name, _ in pkgutil.walk_packages(
        estimators_pkg.__path__, prefix=f"{estimators_pkg.__name__}."
    ):
        if ".tests" in module_name:
            continue

        module = importlib.import_module(module_name)

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, BaseAeonEstimator)
                and not cls.__name__.startswith("_")
                and cls.__name__ not in _SKIP_ESTIMATORS
                and cls.__module__ == module.__name__
            ):
                estimators.append(cls)

    return estimators


@parametrize_with_checks(_get_estimators())
def test_aeon_estimator(check):
    """Run aeon estimator checks on all public aeon estimators."""
    check()
