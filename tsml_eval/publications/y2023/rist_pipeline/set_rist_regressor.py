"""Regressors used in the publication."""

__author__ = ["MatthewMiddlehurst"]

from tsml_eval.utils.functions import str_in_nested_list

rist_regressors = [
    ["inceptione", "inception-e", "inceptiontime", "InceptionTimeRegressor"],
    ["rocket", "RocketRegressor"],
    ["DrCIF", "drcifregressor"],
    ["fresh-prince", "freshprince", "FreshPRINCERegressor"],
    "RDST",
    ["RIST", "rist-extrat"],
    "rist-rf",
    "rist-ridgecv",
]


def _set_rist_regressor(
    regressor_name,
    random_state=None,
    n_jobs=1,
):
    r = regressor_name.lower()

    if not str_in_nested_list(rist_regressors, r):
        raise Exception("UNKNOWN CLASSIFIER ", r, " in set_rist_regressor")

    if (
        r == "inceptione"
        or r == "inception-e"
        or r == "inceptiontime"
        or r == "inceptiontimeregressor"
    ):
        from tsml_eval.estimators.regression.deep_learning import InceptionTimeRegressor

        return InceptionTimeRegressor(random_state=random_state)
    elif r == "rocket" or r == "rocketregressor":
        from aeon.regression.convolution_based import RocketRegressor

        return RocketRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif r == "drcif" or r == "drcifregressor":
        from tsml_eval.estimators.regression.interval_based import DrCIF

        return DrCIF(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif r == "fresh-prince" or r == "freshprince" or r == "freshprinceregressor":
        from aeon.regression.feature_based import FreshPRINCERegressor

        return FreshPRINCERegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif r == "rdst":
        pass
    elif r == "rist" or r == "rist-extrat":
        pass
    elif r == "rist-rf":
        pass
    elif r == "rist-ridgecv":
        pass
