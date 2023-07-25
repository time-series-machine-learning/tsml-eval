"""Regressors used in the publication."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst", "dguijo"]

import numpy as np

from tsml_eval.utils.functions import str_in_nested_list

expansion_regressors = [
    ["1nn-dtw", "KNeighborsTimeSeriesRegressor"],
    "1nn-ed",
    "5nn-dtw",
    "5nn-ed",
    ["fcnn", "fcn", "fcnnregressor", "FCNRegressor"],
    ["FPCARegressor", "fpcregressor", "fpcr"],
    ["fpcar-b-spline", "fpcr-b-spline", "fpcr-bs"],
    ["grid-svr", "grid-supportvectorregressor"],
    [
        "inception",
        "singleinception",
        "individualinception",
        "IndividualInceptionTimeRegressor",
    ],
    ["inceptione", "inception-e", "inceptiontime", "InceptionTimeRegressor"],
    ["rf", "randf", "randomforest", "RandomForestRegressor"],
    ["resnet", "ResNetRegressor"],
    ["rocket", "RocketRegressor"],
    ["multirocket", "multirocketregressor"],
    ["xgb", "xgboost", "xgboostregressor", "XGBRegressor"],
    ["cnn", "CNNRegressor"],
    ["RidgeCV", "ridge"],
    ["RotationForestRegressor", "rotf", "rotationforest"],
    ["tsf", "timeseriesforestregressor"],
    ["DrCIF", "drcifregressor"],
    ["fresh-prince", "freshprince", "FreshPRINCERegressor"],
]


def _set_tser_exp_regressor(
    regressor_name,
    random_state=None,
    n_jobs=1,
):
    r = regressor_name.lower()

    if not str_in_nested_list(expansion_regressors, r):
        raise Exception("UNKNOWN REGRESSOR ", r, " in set_expansion_regressor")

    if r == "1nn-dtw" or r == "kneighborstimeseriesregressor":
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            n_neighbors=1,
            distance="dtw",
            distance_params={"window": 0.1},
        )
    elif r == "1nn-ed":
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            distance="euclidean",
            n_neighbors=1,
        )
    elif r == "5nn-dtw":
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            n_neighbors=5,
            distance="dtw",
            distance_params={"window": 0.1},
        )
    elif r == "5nn-ed":
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            distance="euclidean",
            n_neighbors=5,
        )
    elif r == "fcnn" or r == "fcn" or r == "fcnnregressor" or r == "fcnregressor":
        from tsml_eval.estimators.regression.deep_learning import FCNRegressor

        return FCNRegressor(random_state=random_state)
    elif r == "fpcaregressor" or r == "fpcregressor" or r == "fpcr":
        from tsml.feature_based import FPCARegressor

        return FPCARegressor(n_jobs=n_jobs)
    elif r == "fpcar-b-spline" or r == "fpcr-b-spline" or r == "fpcr-bs":
        from tsml.feature_based import FPCARegressor

        return FPCARegressor(n_jobs=n_jobs, bspline=True, order=4, n_basis=10)
    elif r == "grid-svr" or r == "grid-supportvectorregressor":
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVR

        param_grid = [
            {
                "kernel": ["rbf", "sigmoid"],
                "C": [0.1, 1, 10, 100],
                "gamma": [0.001, 0.01, 0.1, 1],
            }
        ]

        return GridSearchCV(
            SVR(), param_grid, scoring="neg_mean_squared_error", cv=3, n_jobs=n_jobs
        )
    elif (
        r == "inception"
        or r == "singleinception"
        or r == "individualinception"
        or r == "individualinceptiontimeregressor"
    ):
        from tsml_eval.estimators.regression.deep_learning import (
            IndividualInceptionTimeRegressor,
        )

        return IndividualInceptionTimeRegressor(random_state=random_state)
    elif (
        r == "inceptione"
        or r == "inception-e"
        or r == "inceptiontime"
        or r == "inceptiontimeregressor"
    ):
        from tsml_eval.estimators.regression.deep_learning import InceptionTimeRegressor

        return InceptionTimeRegressor(random_state=random_state)
    elif (
        r == "rf" or r == "randf" or r == "randomforest" or r == "randomforestregressor"
    ):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=500,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif r == "resnet" or r == "resnetregressor":
        from tsml_eval.estimators.regression.deep_learning import ResNetRegressor

        return ResNetRegressor(random_state=random_state)
    elif r == "rocket" or r == "rocketregressor":
        from aeon.regression.convolution_based import RocketRegressor

        return RocketRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif r == "multirocket" or r == "multirocketregressor":
        from aeon.regression.convolution_based import RocketRegressor

        return RocketRegressor(
            rocket_transform="multirocket",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif r == "xgb" or r == "xgboost" or r == "xgboostregressor" or r == "xgbregressor":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=500,
            n_jobs=n_jobs,
            learning_rate=0.1,
            random_state=random_state,
        )
    elif r == "cnn" or r == "cnnregressor":
        from tsml_eval.estimators.regression.deep_learning import CNNRegressor

        return CNNRegressor(random_state=random_state)
    elif r == "ridgecv" or r == "ridge":
        from sklearn.linear_model import RidgeCV

        return RidgeCV(
            fit_intercept=True,
            alphas=np.logspace(-3, 3, 10),
        )
    elif r == "rotationforestregressor" or r == "rotf" or r == "rotationforest":
        from aeon.regression.sklearn import RotationForestRegressor

        return RotationForestRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif r == "tsf" or r == "timeseriesforestregressor":
        from aeon.regression.interval_based import TimeSeriesForestRegressor

        from tsml_eval.estimators.regression.column_ensemble import (
            ColumnEnsembleRegressor,
        )

        estimators = [
            (
                "tsf",
                TimeSeriesForestRegressor(
                    random_state=random_state, n_estimators=500, n_jobs=n_jobs
                ),
                None,
            )
        ]
        return ColumnEnsembleRegressor(estimators)
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
