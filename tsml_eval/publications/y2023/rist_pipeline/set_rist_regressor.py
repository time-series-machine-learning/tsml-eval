"""Regressors used in the publication."""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np

from tsml_eval.utils.functions import str_in_nested_list

rist_regressors = [
    ["inceptione", "inception-e", "inceptiontime", "InceptionTimeRegressor"],
    ["rocket", "RocketRegressor"],
    ["drcif", "DrCIFRegressor"],
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
    **kwargs,
):
    r = regressor_name.lower()

    if not str_in_nested_list(rist_regressors, r):
        raise ValueError(f"UNKNOWN REGRESSOR: {r} in _set_rist_regressor")

    if (
        r == "inceptione"
        or r == "inception-e"
        or r == "inceptiontime"
        or r == "inceptiontimeregressor"
    ):
        from aeon.regression.deep_learning import InceptionTimeRegressor

        return InceptionTimeRegressor(random_state=random_state, **kwargs)
    elif r == "rocket" or r == "rocketregressor":
        from aeon.regression.convolution_based import RocketRegressor

        return RocketRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif r == "drcif" or r == "drcifregressor":
        from aeon.regression.interval_based import DrCIFRegressor

        return DrCIFRegressor(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif r == "fresh-prince" or r == "freshprince" or r == "freshprinceregressor":
        from aeon.regression.feature_based import FreshPRINCERegressor

        return FreshPRINCERegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "rdstregressor" or r == "rdst":
        from aeon.regression.shapelet_based import RDSTRegressor

        return RDSTRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif r == "ristregressor" or r == "rist" or r == "rist-extrat":
        from aeon.regression.hybrid import RISTRegressor
        from sklearn.ensemble import ExtraTreesRegressor

        return RISTRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=ExtraTreesRegressor(n_estimators=500),
            **kwargs,
        )
    elif r == "rist-rf":
        from aeon.regression.hybrid import RISTRegressor
        from sklearn.ensemble import RandomForestRegressor

        return RISTRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=RandomForestRegressor(n_estimators=500),
            **kwargs,
        )
    elif r == "rist-ridgecv":
        from aeon.regression.hybrid import RISTRegressor
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return RISTRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=make_pipeline(
                StandardScaler(with_mean=False),
                RidgeCV(alphas=np.logspace(-4, 4, 20)),
            ),
            **kwargs,
        )
