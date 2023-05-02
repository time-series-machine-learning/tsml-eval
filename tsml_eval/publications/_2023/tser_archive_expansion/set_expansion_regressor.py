# -*- coding: utf-8 -*-

__author__ = ["TonyBagnall", "MatthewMiddlehurst", "dguijo"]

import numpy as np


def _set_expansion_regressor(
    regressor_name,
    random_state=None,
    n_jobs=1,
):
    r = regressor_name.lower()

    if r == "cnn" or r == "cnnregressor":
        from sktime.regression.deep_learning.cnn import CNNRegressor

        return CNNRegressor(random_state=random_state)
    elif r == "resnet" or r == "resnetregressor":
        from tsml_eval.estimators.regression.deep_learning import ResNetRegressor

        return ResNetRegressor(random_state=random_state)
    elif r == "inception" or r == "inceptiontime" or r == "inceptiontimeregressor":
        from tsml_eval.estimators.regression.deep_learning import InceptionTimeRegressor

        return InceptionTimeRegressor(random_state=random_state)
    elif r == "singleinception" or r == "individualinception":
        from tsml_eval.estimators.regression.deep_learning import (
            IndividualInceptionTimeRegressor,
        )

        return IndividualInceptionTimeRegressor(random_state=random_state)
    elif r == "fcnn" or r == "fcn" or r == "fcnnregressor":
        from tsml_eval.estimators.regression.deep_learning import FCNRegressor

        return FCNRegressor(random_state=random_state)
    elif r == "1nn-ed":
        from tsml_eval.estimators.regression.distance_based import (
            KNeighborsTimeSeriesRegressor,
        )

        return KNeighborsTimeSeriesRegressor(
            distance="euclidean",
            n_neighbours=1,
        )
    elif r == "5nn-ed":
        from tsml_eval.estimators.regression.distance_based import (
            KNeighborsTimeSeriesRegressor,
        )

        return KNeighborsTimeSeriesRegressor(
            distance="euclidean",
            n_neighbours=5,
        )
    elif r == "1nn-dtw":
        from tsml_eval.estimators.regression.distance_based import (
            KNeighborsTimeSeriesRegressor,
        )

        return KNeighborsTimeSeriesRegressor(
            n_neighbours=1,
            distance="dtw",
            distance_params={"window": 0.1},
        )
    elif r == "5nn-dtw":
        from tsml_eval.estimators.regression.distance_based import (
            KNeighborsTimeSeriesRegressor,
        )

        return KNeighborsTimeSeriesRegressor(
            n_neighbours=5,
            distance="dtw",
            distance_params={"window": 0.1},
        )
    elif r == "rocket" or r == "rocketregressor":
        from sktime.regression.kernel_based import RocketRegressor

        return RocketRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif r == "multirocket" or r == "multirocketregressor":
        from sktime.regression.kernel_based import RocketRegressor

        return RocketRegressor(
            rocket_transform="multirocket",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif r == "tsf":
        from sktime.regression.interval_based import TimeSeriesForestRegressor

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
    elif r == "fresh-prince" or r == "freshprince":
        from tsml_eval.estimators.regression.featured_based import FreshPRINCERegressor

        return FreshPRINCERegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif r == "drcif":
        from tsml_eval.estimators.regression.interval_based import DrCIF

        return DrCIF(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif r == "rotf" or r == "rotationforest":
        from tsml_eval.estimators.regression.sklearn import (
            RotationForest,
            SklearnToTsmlRegressor,
        )

        model_params = {
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

        return SklearnToTsmlRegressor(RotationForest(**model_params))
    elif r == "lr" or r == "linearregression":
        from sklearn.linear_model import LinearRegression

        from tsml_eval.estimators import SklearnToTsmlRegressor

        model_params = {"fit_intercept": True, "n_jobs": n_jobs}

        return SklearnToTsmlRegressor(LinearRegression(**model_params))
    elif r == "ridgecv" or r == "ridge":
        from sklearn.linear_model import RidgeCV

        from tsml_eval.estimators import SklearnToTsmlRegressor

        model_params = {"fit_intercept": True, "alphas": np.logspace(-3, 3, 10)}

        return SklearnToTsmlRegressor(RidgeCV(**model_params))
    elif r == "grid-svr" or r == "grid-supportvectorregressor":
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVR

        from tsml_eval.estimators import SklearnToTsmlRegressor

        param_grid = [
            {
                "kernel": ["rbf", "sigmoid"],
                "C": [0.1, 1, 10, 100],
                "gamma": [0.001, 0.01, 0.1, 1],
            }
        ]

        scoring = "neg_mean_squared_error"

        return SklearnToTsmlRegressor(
            GridSearchCV(SVR(), param_grid, scoring=scoring, n_jobs=n_jobs, cv=3)
        )
    elif r == "rf" or r == "randomforest":
        from sklearn.ensemble import RandomForestRegressor

        from tsml_eval.estimators import SklearnToTsmlRegressor

        model_params = {
            "n_estimators": 500,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }

        return SklearnToTsmlRegressor(RandomForestRegressor(**model_params))
    elif r == "xgb" or r == "xgboost":
        from xgboost import XGBRegressor

        from tsml_eval.estimators import SklearnToTsmlRegressor

        model_params = {
            "n_estimators": 500,
            "n_jobs": n_jobs,
            "learning_rate": 0.1,
            "random_state": random_state,
        }

        return SklearnToTsmlRegressor(XGBRegressor(**model_params))
    elif r == "fpcr":
        from tsml_eval.estimators.regression.sofr.fpcr import FPCRegressor

        return FPCRegressor(n_components=10)

    elif r == "fpcr-b-spline":
        from tsml_eval.estimators.regression.sofr.fpcr import FPCRegressor

        model_params = {
            "smooth": "B-spline",
            "order": 4,
            "n_components": 10,
            "n_basis": 10,
        }

        return FPCRegressor(**model_params)

    # invalid regressor
    else:
        raise Exception("UNKNOWN REGRESSOR ", r, " in set_regressor")
