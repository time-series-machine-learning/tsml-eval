"""Regressors used in the publication."""

__author__ = ["MatthewMiddlehurst"]

import numpy as np

from tsml_eval.utils.functions import str_in_nested_list

rist_regressors = [
    ["inceptione", "inception-e", "inceptiontime", "InceptionTimeRegressor"],
    ["rocket", "RocketRegressor"],
    ["DrCIF", "drcifregressor"],
    ["fresh-prince", "freshprince", "FreshPRINCERegressor"],
    ["RDSTRegressor", "rdst"],
    ["RISTRegressor", "rist", "rist-extrat"],
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
    elif r == "rdstregressor" or r == "rdst":
        from tsml.shapelet_based import RDSTRegressor

        return RDSTRegressor(random_state=random_state, n_jobs=n_jobs)
    elif r == "ristregressor" or r == "rist" or r == "rist-extrat":
        from sklearn.ensemble import ExtraTreesRegressor
        from tsml.hybrid import RISTRegressor

        return RISTRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=ExtraTreesRegressor(n_estimators=500),
        )
    elif r == "rist-rf":
        from sklearn.ensemble import RandomForestRegressor
        from tsml.hybrid import RISTRegressor

        return RISTRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=RandomForestRegressor(n_estimators=500),
        )
    elif r == "rist-ridgecv":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from tsml.hybrid import RISTRegressor

        return RISTRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=make_pipeline(
                StandardScaler(with_mean=False), RidgeCV(alphas=np.logspace(-4, 4, 20))
            ),
        )
