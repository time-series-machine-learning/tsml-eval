# -*- coding: utf-8 -*-
"""Set regressor function."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
from sklearn.pipeline import make_pipeline


def set_regressor(
    regressor_name,
    random_state=None,
    build_train_file=False,
    n_jobs=1,
    fit_contract=0,
    kwargs=None,
):
    """Return a regressor matching a given input name.

    Basic way of creating a regressor to build using the default or alternative
    settings. This set up is to help with batch jobs for multiple problems and to
    facilitate easy reproducibility through run_regression_experiment.

    Generally, inputting a regressor class name will return said regressor with
    default settings.

    Parameters
    ----------
    regressor_name : str
        String indicating which regressor to be returned.
    random_state : int, RandomState instance or None, default=None
        Random seed or RandomState object to be used in the regressor if available.
    build_train_file : bool, default=False
        Whether a train data results file is being produced. If True, regressor specific
        parameters for generating train results will be toggled if available.
    n_jobs: int, default=1
        The number of jobs to run in parallel for both regressor ``fit`` and
        ``predict`` if available. `-1` means using all processors.
    fit_contract: int, default=0
        Contract time in minutes for regressor ``fit`` if available.

    Return
    ------
    regressor: A BaseRegressor.
        The regressor matching the input regressor name.
    """
    r = regressor_name.lower()

    if r == "cnn" or r == "cnnregressor":
        from sktime.regression.deep_learning.cnn import CNNRegressor

        return CNNRegressor(random_state=random_state)

    elif r == "tapnet" or r == "tapnetregressor":
        from sktime.regression.deep_learning.tapnet import TapNetRegressor

        return TapNetRegressor(random_state=random_state)
    # resnet implementation is we think buggy do not use yet
    #    elif r == "resnet" or r == "resnetregressor":
    #        from tsml_eval.sktime_estimators.regression.deep_learning import
    #        ResNetRegressor
    #
    #        return ResNetRegressor(random_state=random_state)

    elif r == "inception" or r == "inceptiontime" or r == "inceptiontimeregressor":
        from tsml_eval.sktime_estimators.regression.deep_learning import (
            InceptionTimeRegressor,
        )

        return InceptionTimeRegressor(random_state=random_state)

    elif r == "fcnn" or r == "fcn" or r == "fcnnregressor":
        from tsml_eval.sktime_estimators.regression.deep_learning import FCNRegressor

        return FCNRegressor(random_state=random_state)

    elif r == "sktime-1nn-ed":
        from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            n_neighbors=1,
            distance="euclidean",
        )
    elif r == "sktime-1nn-dtw":
        from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            distance="dtw",
            distance_params={"window": 0.1},
            n_neighbors=1,
            n_jobs=n_jobs,
        )
    elif r == "sktime-5nn-ed":
        from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            n_neighbors=5,
            distance="euclidean",
        )
    elif r == "sktime-5nn-dtw":
        from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            distance="dtw",
            n_neighbors=5,
            distance_params={"window": 0.1},
            n_jobs=n_jobs,
        )
    elif r == "1nn-ed":
        from tsml_eval.sktime_estimators.regression.distance_based import (
            KNeighborsTimeSeriesRegressor,
        )

        return KNeighborsTimeSeriesRegressor(
            distance="euclidean",
            n_neighbours=1,
        )
    elif r == "5nn-ed":
        from tsml_eval.sktime_estimators.regression.distance_based import (
            KNeighborsTimeSeriesRegressor,
        )

        return KNeighborsTimeSeriesRegressor(
            distance="euclidean",
            n_neighbours=5,
        )
    elif r == "1nn-dtw":
        from tsml_eval.sktime_estimators.regression.distance_based import (
            KNeighborsTimeSeriesRegressor,
        )

        return KNeighborsTimeSeriesRegressor(
            n_neighbours=1,
            distance="dtw",
            distance_params={"window": 0.1},
        )
    elif r == "5nn-dtw":
        from tsml_eval.sktime_estimators.regression.distance_based import (
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
    elif r == "minirocket" or r == "minirocketregressor":
        from sktime.regression.kernel_based import RocketRegressor

        return RocketRegressor(
            rocket_transform="minirocket",
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
    elif r == "hydra" or r == "hydraregressor":
        from tsml_eval.sktime_estimators.regression.convolution_based import (
            HydraRegressor,
        )

        return HydraRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif r == "tsf" or r == "timeseriesforestregressor":
        from sktime.regression.interval_based import TimeSeriesForestRegressor

        return TimeSeriesForestRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
        )

    # Other
    elif r == "dummy" or r == "dummyregressor":
        # todo we need an actual dummy for this. use tiny rocket for testing purposes
        #  currently
        from sktime.regression.kernel_based import RocketRegressor

        return RocketRegressor(
            num_kernels=50,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        # raise ValueError(f" Regressor {name} is not avaiable")

    # regression package regressors
    elif r == "fresh-prince" or r == "freshprince":
        from tsml_eval.sktime_estimators.regression.featured_based import (
            FreshPRINCERegressor,
        )

        return FreshPRINCERegressor(
            n_estimators=500,
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
        )

    elif r == "drcif":
        from tsml_eval.sktime_estimators.regression.interval_based import DrCIF

        return DrCIF(
            n_estimators=500,
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
        )
    elif r == "stc" or r == "str":
        from tsml_eval.sktime_estimators.regression.shapelet_based import (
            ShapeletTransformRegressor,
        )

        return ShapeletTransformRegressor(
            transform_limit_in_minutes=120,
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
        )
    elif r == "str-default":
        from tsml_eval.sktime_estimators.regression.shapelet_based import (
            ShapeletTransformRegressor,
        )

        return ShapeletTransformRegressor(
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
        )
    elif r == "str-ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        from tsml_eval.sktime_estimators.regression.shapelet_based import (
            ShapeletTransformRegressor,
        )

        return ShapeletTransformRegressor(
            estimator=make_pipeline(
                StandardScaler(with_mean=False),
                RidgeCV(alphas=np.logspace(-3, 3, 10)),
            ),
            transform_limit_in_minutes=120,
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
        )
    elif r == "tde":
        from tsml_eval.sktime_estimators.regression.dictionary_based import (
            TemporalDictionaryEnsemble,
        )

        return TemporalDictionaryEnsemble(
            random_state=random_state,
            save_train_predictions=build_train_file,
            n_jobs=n_jobs,
        )
    elif r == "arsenal":
        from tsml_eval.sktime_estimators.regression.convolution_based import Arsenal

        return Arsenal(
            random_state=random_state,
            save_transformed_data=build_train_file,
            n_jobs=n_jobs,
        )
    elif r == "hc2" or r == "hivecotev2":
        from tsml_eval.sktime_estimators.regression.hybrid import HIVECOTEV2

        return HIVECOTEV2(random_state=random_state, n_jobs=n_jobs)

    # sklearn regressors
    elif r == "rotf" or r == "rotationforest":
        from tsml_eval.sktime_estimators.regression.sklearn import (
            RotationForest,
            SklearnBaseRegressor,
        )

        model_params = {
            "random_state": random_state,
            "save_transformed_data": build_train_file,
            "n_jobs": n_jobs,
        }

        return SklearnBaseRegressor(RotationForest(**model_params))

    elif r == "lr" or r == "linearregression":
        from sklearn.linear_model import LinearRegression

        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor

        model_params = {"fit_intercept": True, "n_jobs": n_jobs}

        return SklearnBaseRegressor(LinearRegression(**model_params))

    elif r == "ridgecv" or r == "ridge":
        from sklearn.linear_model import RidgeCV

        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor

        model_params = {"fit_intercept": True, "alphas": np.logspace(-3, 3, 10)}

        return SklearnBaseRegressor(RidgeCV(**model_params))

    elif r == "svr" or r == "supportvectorregressor":
        from sklearn.svm import SVR

        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor

        model_params = {"kernel": "rbf", "C": 1}

        return SklearnBaseRegressor(SVR(**model_params))
    elif r == "grid-svr" or r == "grid-supportvectorregressor":
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVR

        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor

        param_grid = [
            {
                "kernel": ["rbf", "sigmoid"],
                "C": [0.1, 1, 10, 100],
                "gamma": [0.001, 0.01, 0.1, 1],
            }
        ]

        scoring = "neg_mean_squared_error"

        return SklearnBaseRegressor(
            GridSearchCV(SVR(), param_grid, scoring=scoring, n_jobs=n_jobs, cv=3)
        )
    elif r == "rf" or r == "randomforest":
        from sklearn.ensemble import RandomForestRegressor

        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor

        model_params = {
            "n_estimators": 100,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }

        return SklearnBaseRegressor(RandomForestRegressor(**model_params))

    elif r == "xgb" or r == "xgboost":
        from xgboost import XGBRegressor  # pip install xgboost

        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor

        model_params = {
            "n_estimators": 100,
            "n_jobs": n_jobs,
            "learning_rate": 0.1,
            "random_state": random_state,
        }

        return SklearnBaseRegressor(XGBRegressor(**model_params))

    # invalid regressor
    else:
        raise Exception("UNKNOWN REGRESSOR ", r, " in set_regressor")
