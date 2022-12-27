# -*- coding: utf-8 -*-
"""Set regressor function."""
__author__ = ["TonyBagnall"]

import numpy as np
from sklearn.pipeline import make_pipeline


def set_regressor(regressor, resample_id=None, train_file=False, n_jobs=1, kwargs=None):
    """Construct a regressor, possibly seeded for reproducibility.

    Basic way of creating the regressor to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducibility for use with load_and_run_regression_experiment. You can pass a
    regression object instead to run_regression_experiment.
    TODO: add threads, contract and checkpoint options

    Parameters
    ----------
    regressor : str
        String indicating which Regressor you want.
    resample_id : int or None, default=None
        Regressor random seed.
    train_file : bool, default=False
        Whether a train file is being produced.
    n_jobs: for threading

    Return
    ------
    regressor: A BaseRegressor.
        The regressor matching the input regressor name.
    """
    name = regressor.lower()
    if name == "cnn" or name == "cnnregressor":
        from sktime.regression.deep_learning.cnn import CNNRegressor

        return CNNRegressor(random_state=resample_id)
    elif name == "tapnet" or name == "tapnetregressor":
        from sktime.regression.deep_learning.tapnet import TapNetRegressor

        return TapNetRegressor(random_state=resample_id)
    elif name == "knn" or name == "kneighborstimeseriesregressor":
        from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            # random_state=resample_id,
            n_jobs=n_jobs,
        )
    elif name == "1nn-ed":
        from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
        
        return KNeighborsTimeSeriesRegressor(
            n_neighbors=1,
            distance='euclidean',
            # random_state=resample_id,
            n_jobs=n_jobs,
        )
    elif name == "1nn-dtw":
        from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            n_neighbors=1,
            distance='dtw',
            metric_params={'window': kwargs['n_ts']*0.1},
            # random_state=resample_id,
            n_jobs=n_jobs,
        )
    elif name == "5nn-ed":
        from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            n_neighbors=5,
            distance='euclidean',
            # random_state=resample_id,
            n_jobs=n_jobs,
        )
    elif name == "5nn-dtw":
        from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            n_neighbors=5,
            distance='dtw',
            metric_params={'window': kwargs['n_ts']*0.1},
            # random_state=resample_id,
            n_jobs=n_jobs,
        )
    elif name == "rocket" or name == "rocketregressor":
        from sktime.regression.kernel_based import RocketRegressor

        return RocketRegressor(
            random_state=resample_id,
            n_jobs=n_jobs,
        )
    
    elif name == "minirocket" or name == "minirocketregressor":
        from sktime.regression.kernel_based import RocketRegressor
        return RocketRegressor(
            rocket_transform="minirocket",
            random_state=resample_id,
            n_jobs=n_jobs,
        )

    elif name == "multirocket" or name == "multirocketregressor":
        from sktime.regression.kernel_based import RocketRegressor

        return RocketRegressor(
            rocket_transform="multirocket",
            random_state=resample_id,
            n_jobs=n_jobs,
        )
    elif name == "tsf" or name == "timeseriesforestregressor":
        from sktime.regression.interval_based import TimeSeriesForestRegressor

        return TimeSeriesForestRegressor(
            random_state=resample_id,
            n_jobs=n_jobs,
        )
    
    elif name == "hydra" or name == "hydraregressor":
        from tsml_eval.sktime_estimators.regression.convolution_based import HydraRegressor

        return HydraRegressor(
            random_state=resample_id,
            n_jobs=n_jobs,
        )
        
    # Other
    elif name == "dummy" or name == "dummyregressor":
        # todo we need an actual dummy for this
        raise ValueError(f" Regressor {name} is not avaiable")

    # regression package regressors
    elif name == "drcif":
        from tsml_eval.sktime_estimators.regression.interval_based import DrCIF

        return DrCIF(
            n_estimators=500,
            random_state=resample_id,
            save_transformed_data=train_file,
            n_jobs=n_jobs,
        )
    elif name == "stc" or name == "str":
        from tsml_eval.sktime_estimators.regression.shapelet_based import (
            ShapeletTransformRegressor,
        )

        return ShapeletTransformRegressor(
            transform_limit_in_minutes=120,
            random_state=resample_id,
            save_transformed_data=train_file,
            n_jobs=n_jobs,
        )
    elif name == "str-default":
        from tsml_eval.sktime_estimators.regression.shapelet_based import (
            ShapeletTransformRegressor,
        )

        return ShapeletTransformRegressor(
            random_state=resample_id,
            save_transformed_data=train_file,
            n_jobs=n_jobs,
        )
    elif name == "str-ridge":
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
            random_state=resample_id,
            save_transformed_data=train_file,
            n_jobs=n_jobs,
        )
    elif name == "tde":
        from tsml_eval.sktime_estimators.regression.dictionary_based import (
            TemporalDictionaryEnsemble,
        )

        return TemporalDictionaryEnsemble(
            random_state=resample_id, save_train_predictions=train_file, n_jobs=n_jobs
        )
    elif name == "arsenal":
        from tsml_eval.sktime_estimators.regression.convolution_based import Arsenal

        return Arsenal(
            random_state=resample_id, save_transformed_data=train_file, n_jobs=n_jobs
        )
    elif name == "hc2" or name == "hivecotev2":
        from tsml_eval.sktime_estimators.regression.hybrid import HIVECOTEV2

        return HIVECOTEV2(random_state=resample_id, n_jobs=n_jobs)

    # sklearn regerssors
    # todo experiments for these
    elif name == "rotf" or name == "rotationforest":
        from tsml_eval.sktime_estimators.regression.sklearn import RotationForest

        return RotationForest(
            random_state=resample_id, save_transformed_data=train_file, n_jobs=n_jobs
        )

    elif name == "lr" or name == "linearregression":
        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor
        from sklearn.linear_model import LinearRegression

        model_params = {"fit_intercept": True,
                        "n_jobs": n_jobs}

        return SklearnBaseRegressor(LinearRegression(**model_params))
    
    elif name == "ridgecv" or name == "ridge":
        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor
        from sklearn.linear_model import RidgeCV

        model_params = {"fit_intercept": True,
                        "alphas": np.logspace(-3, 3, 10)}

        return SklearnBaseRegressor(RidgeCV(**model_params))

    elif name == "svr" or name == "supportvectorregressor":
        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor
        from sklearn.svm import SVR

        model_params = {"kernel": 'rbf',
                        "C": 1}

        return SklearnBaseRegressor(SVR(**model_params))

    elif name == "grid-svr" or name == "grid-supportvectorregressor":
        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor
        from sklearn.svm import SVR
        from sklearn.model_selection import GridSearchCV

        param_grid = [{
            "kernel": ['rbf', 'sigmoid'], 
            "C": [0.1, 1, 10, 100], 
            "gamma": [0.001, 0.01, 0.1, 1]
            }]
        
        scoring = 'neg_mean_squared_error'

        return SklearnBaseRegressor(GridSearchCV(
                SVR(), 
                param_grid, 
                scoring=scoring, 
                n_jobs=n_jobs, 
                cv=3))
    
    elif name == "rf" or name == "randomforest":
        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor
        from sklearn.ensemble import RandomForestRegressor

        model_params = {"n_estimators": 100,
                        "n_jobs": n_jobs,
                        "random_state": resample_id}

        return SklearnBaseRegressor(RandomForestRegressor(**model_params))
        
    elif name == "xgb" or name == "xgboost":
        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor
        from xgboost import XGBRegressor # pip install xgboost

        model_params = {"n_estimators": 100,
                        "n_jobs": n_jobs,
                        "learning_rate": 0.1,
                        "random_state": resample_id}

        return SklearnBaseRegressor(XGBRegressor(**model_params))

    else:
        raise ValueError(f" Regressor {name} is not avaiable")
