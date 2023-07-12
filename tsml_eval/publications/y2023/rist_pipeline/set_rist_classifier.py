"""Classifiers used in the publication."""

__author__ = ["MatthewMiddlehurst"]

from tsml_eval.utils.functions import str_in_nested_list

rist_classifiers = [
    ["FreshPRINCEClassifier", "freshprince"],
    ["ShapeletTransformClassifier", "stc", "stc-2hour"],
    "RDST",
    ["RSTSFClassifier", "rstsf", "r-stsf"],
    "DrCIF",
    ["RocketClassifier", "rocket"],
    ["HIVECOTEV2", "hc2"],
    ["RIST", "rist-extrat"],
    "rist-rf",
    "rist-ridgecv",
    "intervalpipeline",
]


def _set_rist_classifier(
    classifier_name,
    random_state=None,
    n_jobs=1,
):
    c = classifier_name.lower()

    if not str_in_nested_list(rist_classifiers, c):
        raise Exception("UNKNOWN CLASSIFIER ", c, " in set_rist_classifier")

    if c == "freshprinceclassifier" or c == "freshprince":
        from tsml_eval.estimators.classification.feature_based import (
            FreshPRINCEClassifier,
        )

        return FreshPRINCEClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "shapelettransformclassifier" or c == "stc" or c == "stc-2hour":
        from tsml_eval.estimators.classification.shapelet_based import (
            ShapeletTransformClassifier,
        )

        return ShapeletTransformClassifier(
            transform_limit_in_minutes=120,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "rdst":
        from tsml_eval.estimators.classification.shapelet_based.rdst import RDST

        return RDST(random_state=random_state)
    elif c == c == "rstsfclassifier" or c == "rstsf" or c == "r-stsf":
        from tsml.interval_based import RSTSFClassifier

        return RSTSFClassifier(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs
        )
    elif c == "drcif":
        from aeon.classification.interval_based import DrCIF

        return DrCIF(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "rocketclassifier" or c == "rocket":
        from aeon.classification.convolution_based import RocketClassifier

        return RocketClassifier(random_state=random_state, n_jobs=n_jobs)
    elif c == "hivecotev2" or c == "hc2":
        from aeon.classification.hybrid import HIVECOTEV2

        return HIVECOTEV2(
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif c == "rist" or c == "rist-extrat":
        pass
    elif c == "rist-rf":
        pass
    elif c == "rist-ridgecv":
        pass
    elif c == "intervalpipeline":
        pass
