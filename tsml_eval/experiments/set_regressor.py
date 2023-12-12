"""Set regressor function."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
from sklearn.pipeline import make_pipeline

from tsml_eval.utils.functions import str_in_nested_list

convolution_based_regressors = [
    ["rocketregressor", "rocket"],
    ["minirocket", "minirocketregressor"],
    ["multirocket", "multirocketregressor"],
    ["hydraregressor", "hydra"],
    ["arsenal", "arsenalregressor"],
]
deep_learning_regressors = [
    ["cnnregressor", "cnn"],
    ["tapnetregressor", "tapnet"],
    ["resnetregressor", "resnet"],
    ["inceptiontimeregressor", "inception", "inceptiontime"],
    ["individualinceptionregressor", "singleinception", "individualinception"],
    ["fcnregressor", "fcnn", "fcn"],
]
dictionary_based_regressors = [
    ["temporaldictionaryensemble", "tde"],
]
distance_based_regressors = [
    "1nn-ed",
    "5nn-ed",
    ["kneighborstimeseriesregressor", "1nn-dtw"],
    "5nn-dtw",
    "1nn-msm",
    "5nn-msm",
]
feature_based_regressors = [
    ["freshprinceregressor", "fresh-prince", "freshprince"],
    "freshprince-500",
    ["fpcaregressor", "fpcregressor", "fpcr"],
    ["fpcar-b-spline", "fpcr-b-spline", "fpcr-bs"],
]
hybrid_regressors = [
    ["hivecotev2", "hc2"],
]
interval_based_regressors = [
    ["timeseriesforestregressor", "tsf"],
    "tsf-i",
    "tsf-500",
    ["drcif", "drcifregressor"],
    "drcif-500",
]
other_regressors = [
    ["dummyregressor", "dummy", "dummyregressor-tsml"],
    "dummyregressor-aeon",
    ["dummyregressor-sklearn", "meanpredictorregressor", "dummymeanpred"],
    ["medianpredictorregressor", "dummymedianpred"],
]
shapelet_based_regressors = [
    "str-2hour",
    ["shapelettransformregressor", "str", "stc"],
    "str-2hour-ridge",
]
vector_regressors = [
    ["rotationforestregressor", "rotf", "rotationforest"],
    ["linearregression", "lr"],
    ["ridgecv", "ridge"],
    ["svr", "svm", "supportvectorregressor"],
    ["grid-svr", "grid-svm", "grid-supportvectorregressor"],
    ["randomforestregressor", "rf", "randomforest"],
    ["randomforest-500", "rf-500"],
    ["xgbregressor", "xgboost"],
    ["xgb-100", "xgboost-100"],
    ["xgb-500", "xgboost-500"],
]


def set_regressor(
    regressor_name,
    random_state=None,
    n_jobs=1,
    build_train_file=False,
    fit_contract=0,
    checkpoint=None,
    **kwargs,
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

    if str_in_nested_list(convolution_based_regressors, r):
        return _set_regressor_convolution_based(
            r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(deep_learning_regressors, r):
        return _set_regressor_deep_learning(
            r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(dictionary_based_regressors, r):
        return _set_regressor_dictionary_based(
            r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(distance_based_regressors, r):
        return _set_regressor_distance_based(
            r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(feature_based_regressors, r):
        return _set_regressor_feature_based(
            r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(hybrid_regressors, r):
        return _set_regressor_hybrid(
            r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(interval_based_regressors, r):
        return _set_regressor_interval_based(
            r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(other_regressors, r):
        return _set_regressor_other(
            r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(shapelet_based_regressors, r):
        return _set_regressor_shapelet_based(
            r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(vector_regressors, r):
        return _set_regressor_vector(
            r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    else:
        raise ValueError(f"UNKNOWN REGRESSOR: {r} in set_regressor")


def _set_regressor_convolution_based(
    r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if r == "rocketregressor" or r == "rocket":
        from aeon.regression.convolution_based import RocketRegressor

        return RocketRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif r == "minirocket" or r == "minirocketregressor":
        from aeon.regression.convolution_based import RocketRegressor

        return RocketRegressor(
            rocket_transform="minirocket",
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "multirocket" or r == "multirocketregressor":
        from aeon.regression.convolution_based import RocketRegressor

        return RocketRegressor(
            rocket_transform="multirocket",
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "hydraregressor" or r == "hydra":
        from tsml_eval.estimators.regression.convolution_based import HydraRegressor

        return HydraRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif r == "arsenal" or r == "arsenalregressor":
        from tsml_eval._wip.hc2_regression.arsenal import Arsenal

        return Arsenal(
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )


def _set_regressor_deep_learning(
    r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if r == "cnnregressor" or r == "cnn":
        from aeon.regression.deep_learning.cnn import CNNRegressor

        return CNNRegressor(random_state=random_state, **kwargs)
    elif r == "tapnetregressor" or r == "tapnet":
        from aeon.regression.deep_learning.tapnet import TapNetRegressor

        return TapNetRegressor(random_state=random_state, **kwargs)
    elif r == "resnetregressor" or r == "resnet":
        from aeon.regression.deep_learning import ResNetRegressor

        return ResNetRegressor(random_state=random_state, **kwargs)
    elif r == "inceptiontimeregressor" or r == "inception" or r == "inceptiontime":
        from aeon.regression.deep_learning.inception_time import InceptionTimeRegressor

        return InceptionTimeRegressor(random_state=random_state, **kwargs)
    elif (
        r == "individualinceptionregressor"
        or r == "singleinception"
        or r == "individualinception"
    ):
        from aeon.regression.deep_learning.inception_time import (
            IndividualInceptionRegressor,
        )

        return IndividualInceptionRegressor(random_state=random_state, **kwargs)
    elif r == "fcnregressor" or r == "fcnn" or r == "fcn":
        from aeon.regression.deep_learning import FCNRegressor

        return FCNRegressor(random_state=random_state, **kwargs)


def _set_regressor_dictionary_based(
    r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if r == "temporaldictionaryensemble" or r == "tde":
        from tsml_eval._wip.hc2_regression.tde import TemporalDictionaryEnsemble

        return TemporalDictionaryEnsemble(
            random_state=random_state,
            n_jobs=n_jobs,
            save_train_predictions=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )


def _set_regressor_distance_based(
    r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if r == "1nn-ed":
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(**kwargs)
    elif r == "5nn-ed":
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(n_neighbors=5, **kwargs)
    elif r == "kneighborstimeseriesregressor" or r == "1nn-dtw":
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            distance="dtw", distance_params={"window": 0.1}, **kwargs
        )
    elif r == "5nn-dtw":
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            n_neighbors=5, distance="dtw", distance_params={"window": 0.1}, **kwargs
        )
    elif r == "1nn-msm":
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            n_neighbors=1,
            distance="msm",
            distance_params={"window": None, "independent": True, "c": 1},
        )
    elif r == "5nn-msm":
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

        return KNeighborsTimeSeriesRegressor(
            n_neighbors=5,
            distance="msm",
            distance_params={"window": None, "independent": True, "c": 1},
        )


def _set_regressor_feature_based(
    r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if r == "freshprinceregressor" or r == "fresh-prince" or r == "freshprince":
        from aeon.regression.feature_based import FreshPRINCERegressor

        return FreshPRINCERegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            **kwargs,
        )
    elif r == "freshprince-500":
        from aeon.regression.feature_based import FreshPRINCERegressor

        return FreshPRINCERegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            **kwargs,
        )
    elif r == "fpcaregressor" or r == "fpcregressor" or r == "fpcr":
        from tsml.feature_based import FPCARegressor

        return FPCARegressor(n_jobs=n_jobs, **kwargs)
    elif r == "fpcar-b-spline" or r == "fpcr-b-spline" or r == "fpcr-bs":
        from tsml.feature_based import FPCARegressor

        return FPCARegressor(n_jobs=n_jobs, bspline=True, order=4, n_basis=10, **kwargs)


def _set_regressor_hybrid(
    r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if r == "hivecotev2" or r == "hc2":
        from tsml_eval._wip.hc2_regression.hivecote_v2 import HIVECOTEV2

        return HIVECOTEV2(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )


def _set_regressor_interval_based(
    r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if r == "timeseriesforestregressor" or r == "tsf":
        from aeon.regression.interval_based import TimeSeriesForestRegressor

        return TimeSeriesForestRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            **kwargs,
        )
    elif r == "tsf-i":
        from aeon.regression.interval_based import TimeSeriesForestRegressor
        from tsml.compose import ChannelEnsembleRegressor

        estimators = [
            (
                "tsf",
                TimeSeriesForestRegressor(
                    random_state=random_state,
                    n_jobs=n_jobs,
                    save_transformed_data=build_train_file,
                ),
                "all-split",
            )
        ]

        return ChannelEnsembleRegressor(estimators, **kwargs)
    elif r == "tsf-500":
        from aeon.regression.interval_based import TimeSeriesForestRegressor

        return TimeSeriesForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            **kwargs,
        )
    elif r == "drcif" or r == "drcifregressor":
        from aeon.regression.interval_based import DrCIFRegressor

        return DrCIFRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif r == "drcif-500":
        from aeon.regression.interval_based import DrCIFRegressor

        return DrCIFRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )


def _set_regressor_other(
    r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if r == "dummy" or r == "dummyregressor" or r == "dummyregressor-tsml":
        from tsml.dummy import DummyRegressor

        return DummyRegressor(**kwargs)
    elif r == "dummyregressor-aeon":
        from aeon.regression import DummyRegressor

        return DummyRegressor(**kwargs)
    elif (
        r == "dummyregressor-sklearn"
        or r == "meanpredictorregressor"
        or r == "dummymeanpred"
    ):
        from sklearn.dummy import DummyRegressor

        return DummyRegressor(**kwargs)
    elif r == "medianpredictorregressor" or r == "dummymedianpred":
        from sklearn.dummy import DummyRegressor

        return DummyRegressor(strategy="median", **kwargs)


def _set_regressor_shapelet_based(
    r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if r == "str-2hour":
        from tsml_eval._wip.hc2_regression.str import ShapeletTransformRegressor

        return ShapeletTransformRegressor(
            transform_limit_in_minutes=120,
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif r == "shapelettransformregressor" or r == "str" or r == "stc":
        from tsml_eval._wip.hc2_regression.str import ShapeletTransformRegressor

        return ShapeletTransformRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif r == "str-2hour-ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        from tsml_eval._wip.hc2_regression.str import ShapeletTransformRegressor

        return ShapeletTransformRegressor(
            estimator=make_pipeline(
                StandardScaler(with_mean=False),
                RidgeCV(alphas=np.logspace(-3, 3, 10)),
            ),
            transform_limit_in_minutes=120,
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )


def _set_regressor_vector(
    r, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if r == "rotationforestregressor" or r == "rotf" or r == "rotationforest":
        from aeon.regression.sklearn import RotationForestRegressor

        return RotationForestRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            save_transformed_data=build_train_file,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif r == "linearregression" or r == "lr":
        from sklearn.linear_model import LinearRegression

        return LinearRegression(fit_intercept=True, n_jobs=n_jobs, **kwargs)
    elif r == "ridgecv" or r == "ridge":
        from sklearn.linear_model import RidgeCV

        return RidgeCV(fit_intercept=True, alphas=np.logspace(-3, 3, 10), **kwargs)
    elif r == "svr" or r == "svm" or r == "supportvectorregressor":
        from sklearn.svm import SVR

        return SVR(kernel="rbf", C=1, **kwargs)
    elif r == "grid-svr" or r == "grid-svm" or r == "grid-supportvectorregressor":
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
            estimator=SVR(),
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            n_jobs=n_jobs,
            cv=3,
            **kwargs,
        )
    elif r == "randomforestregressor" or r == "rf" or r == "randomforest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif r == "rf-500" or r == "randomforest-500":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif r == "xgbregressor" or r == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif r == "xgb-100" or r == "xgboost-100":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "xgb-500" or r == "xgboost-500":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
