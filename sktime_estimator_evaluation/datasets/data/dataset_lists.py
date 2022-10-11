# -*- coding: utf-8 -*-
"""Lists of datasets provided in this directory."""

from sktime.registry import all_estimators

def list_all_multivariate_capable_classifiers():
    """ Return a list of all multivariate capable classifiers in sktime."""
    cls = []
    from sktime.registry import all_estimators
    cls = all_estimators(estimator_types="classifier",
                         filter_tags={"capability:multivariate":True}
                         )
    names = [i for i, _ in cls]
    return names

def list_estimators(estimator_type="classifier", multivariate_only=False,
                    univariate_only=False,
                    dictionary=True):
    """ Return a list of all estimators of given type in sktime."""
    cls = []
    filter_tags = {}
    if multivariate_only:
        filter_tags["capability:multivariate"] = True
    if univariate_only:
        filter_tags["capability:multivariate"] = False
    cls = all_estimators(estimator_types=estimator_type, filter_tags=filter_tags)
    names= [i for i, _ in cls]
    return names


str=list_estimators(estimator_type="classifier")
for s in str:
    print(f"\"{s}\",")

str=list_estimators(estimator_type="clusterer")
for s in str:
    print(f"\"{s}\",")
