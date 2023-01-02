# -*- coding: utf-8 -*-
"""Utilities for searching and printing estimators."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

from sktime.registry import all_estimators


def list_estimators(
    estimator_type="classifier",
    multivariate_only=False,
    univariate_only=False,
):
    """Return a list of all estimators of given type in sktime."""
    filter_tags = {}
    if multivariate_only:
        filter_tags["capability:multivariate"] = True
    if univariate_only:
        filter_tags["capability:multivariate"] = False
    cls = all_estimators(estimator_types=estimator_type, filter_tags=filter_tags)
    names = [i for i, _ in cls]
    return names


if __name__ == "__main__":
    str = list_estimators(estimator_type="classifier")
    print(str)  # noqa: T201
    for s in str:
        print(f'"{s}",')  # noqa: T201
