"""Tests for testing functions and classes."""

import pkgutil

import tsml_eval

# collect all modules
ALL_TSML_EVAL_MODULES = pkgutil.walk_packages(
    tsml_eval.__path__, tsml_eval.__name__ + "."
)
ALL_TSML_EVAL_MODULES = [x[1] for x in ALL_TSML_EVAL_MODULES]

ALL_TSML_EVAL_MODULES_NO_TESTS = [
    x
    for x in ALL_TSML_EVAL_MODULES
    if not any(part == "tests" for part in x.split("."))
]
