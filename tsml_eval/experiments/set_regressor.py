# -*- coding: utf-8 -*-
"""Set regressor function."""
__author__ = ["TonyBagnall"]

import numpy as np
from sklearn.pipeline import make_pipeline


def set_regressor(regressor, resample_id=None, train_file=False, n_jobs=1):
    """Construct a regressor, possibly seeded for reproducability.

    Basic way of creating the regressor to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducibility for use with load_and_run_classification_experiment. You can pass a
    classifier object instead to run_classification_experiment.
    TODO: add threads, contract and checkpoint options

    Parameters
    ----------
    regressor : str
        String indicating which Regressor you want.
    resample_id : int or None, default=None
        Classifier random seed.
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
    elif name == "rocket" or name == "rocketregressor":
        from sktime.regression.kernel_based import RocketRegressor

        return RocketRegressor(
            random_state=resample_id,
            n_jobs=n_jobs,
        )
    elif name == "tsf" or name == "timeseriesforestregressor":
        from sktime.regression.interval_based import TimeSeriesForestRegressor

        return TimeSeriesForestRegressor(
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
        return SklearnBaseRegressor(LinearRegression())
    
    elif name == "ridgecv" or name == "ridge":
        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor
        from sklearn.linear_model import RidgeCV
        return SklearnBaseRegressor(RidgeCV())

    elif name == "svr" or name == "supportvectorregressor":
        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor
        from sklearn.svm import SVR
        return SklearnBaseRegressor(SVR())
    
    elif name == "rf" or name == "randomforest":
        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor
        from sklearn.ensemble import RandomForestRegressor
        return SklearnBaseRegressor(RandomForestRegressor())
        
    elif name == "xgb" or name == "xgboost":
        from tsml_eval.sktime_estimators.regression.sklearn import SklearnBaseRegressor
        from xgboost import XGBRegressor
        return SklearnBaseRegressor(XGBRegressor())

    else:
        raise ValueError(f" Regressor {name} is not avaiable")
