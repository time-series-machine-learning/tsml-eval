"""Set regressor function."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]

import numpy as np

from tsml_eval.utils.functions import str_in_nested_list

convolution_based_regressors = [
    ["rocketregressor", "rocket"],
    ["minirocket", "mini-rocket", "minirocketregressor"],
    ["multirocket", "multi-rocket", "multirocketregressor"],
    ["hydraregressor", "hydra"],
    ["multirockethydraregressor", "multirockethydra", "multirocket-hydra"],
]
deep_learning_regressors = [
    ["timecnnregressor", "timecnn", "cnnregressor", "cnn"],
    ["fcnregressor", "fcnn", "fcn"],
    ["mlpregressor", "mlp"],
    ["encoderregressor", "encoder"],
    ["resnetregressor", "resnet"],
    ["individualinceptionregressor", "singleinception", "individualinception"],
    ["inceptiontimeregressor", "inception", "inceptiontime"],
    ["h-inceptiontimeregressor", "h-inceptiontime"],
    ["litetimeregressor", "litetime"],
    ["individualliteregressor", "individuallite"],
    ["disjointcnnregressor", "disjointcnn"],
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
    "summary-500",
    ["summaryregressor", "summary"],
    "catch22-500",
    ["catch22regressor", "catch22"],
    ["freshprinceregressor", "fresh-prince", "freshprince"],
    "freshprince-500",
    "tsfresh-nofs",
    ["tsfreshregressor", "tsfresh"],
    ["fpcaregressor", "fpcregressor", "fpcr"],
    ["fpcar-b-spline", "fpcr-b-spline", "fpcr-bs"],
]
hybrid_regressors = [
    ["ristregressor", "rist", "rist-extrat"],
]
interval_based_regressors = [
    ["timeseriesforestregressor", "tsf"],
    "tsf-i",
    "tsf-500",
    "rise-500",
    ["randomintervalspectralensembleregressor", "rise"],
    "cif-500",
    ["canonicalintervalforestregressor", "cif"],
    ["drcif", "drcifregressor"],
    "drcif-500",
    "summary-intervals",
    ["randomintervals-500", "catch22-intervals-500"],
    ["randomintervalregressor", "randomintervals", "catch22-intervals"],
    ["quantregressor", "quant"],
]
other_regressors = [
    ["dummyregressor", "dummy", "dummyregressor-aeon"],
    "dummyregressor-tsml",
    ["dummyregressor-sklearn", "meanpredictorregressor", "dummymeanpred"],
    ["medianpredictorregressor", "dummymedianpred"],
]
shapelet_based_regressors = [
    ["rdstregressor", "rdst"],
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


def get_regressor_by_name(
    regressor_name,
    random_state=None,
    n_jobs=1,
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
    n_jobs: int, default=1
        The number of jobs to run in parallel for both regressor ``fit`` and
        ``predict`` if available. `-1` means using all processors.
    fit_contract: int, default=0
        Contract time in minutes for regressor ``fit`` if available.
    checkpoint: str or None, default=None
        Path to a checkpoint file to save the regressor if available. No checkpointing
        if None.
    **kwargs
        Additional keyword arguments to be passed to the regressor.

    Return
    ------
    regressor: A BaseRegressor.
        The regressor matching the input regressor name.
    """
    r = regressor_name.lower()

    if str_in_nested_list(convolution_based_regressors, r):
        return _set_regressor_convolution_based(
            r, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(deep_learning_regressors, r):
        return _set_regressor_deep_learning(
            r, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(distance_based_regressors, r):
        return _set_regressor_distance_based(
            r, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(feature_based_regressors, r):
        return _set_regressor_feature_based(
            r, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(hybrid_regressors, r):
        return _set_regressor_hybrid(
            r, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(interval_based_regressors, r):
        return _set_regressor_interval_based(
            r, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(other_regressors, r):
        return _set_regressor_other(
            r, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(shapelet_based_regressors, r):
        return _set_regressor_shapelet_based(
            r, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(vector_regressors, r):
        return _set_regressor_vector(
            r, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    else:
        raise ValueError(f"UNKNOWN REGRESSOR: {r} in get_regressor_by_name")


def _set_regressor_convolution_based(
    r, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if r == "rocketregressor" or r == "rocket":
        from aeon.regression.convolution_based import RocketRegressor

        return RocketRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif r == "minirocket" or r == "mini-rocket" or r == "minirocketregressor":
        from aeon.regression.convolution_based import MiniRocketRegressor

        return MiniRocketRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "multirocket" or r == "multi-rocket" or r == "multirocketregressor":
        from aeon.regression.convolution_based import MultiRocketRegressor

        return MultiRocketRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "hydraregressor" or r == "hydra":
        from aeon.regression.convolution_based import HydraRegressor

        return HydraRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif (
        r == "multirockethydraregressor"
        or r == "multirockethydra"
        or r == "multirocket-hydra"
    ):
        from aeon.regression.convolution_based import MultiRocketHydraRegressor

        return MultiRocketHydraRegressor(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )


def _set_regressor_deep_learning(
    r, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if r == "timecnnregressor" or r == "timecnn" or r == "cnnregressor" or r == "cnn":
        from aeon.regression.deep_learning import TimeCNNRegressor

        return TimeCNNRegressor(random_state=random_state, **kwargs)
    elif r == "fcnregressor" or r == "fcnn" or r == "fcn":
        from aeon.regression.deep_learning import FCNRegressor

        return FCNRegressor(random_state=random_state, **kwargs)
    elif r == "mlpregressor" or r == "mlp":
        from aeon.regression.deep_learning import MLPRegressor

        return MLPRegressor(random_state=random_state, **kwargs)
    elif r == "encoderregressor" or r == "encoder":
        from aeon.regression.deep_learning import EncoderRegressor

        return EncoderRegressor(random_state=random_state, **kwargs)
    elif r == "resnetregressor" or r == "resnet":
        from aeon.regression.deep_learning import ResNetRegressor

        return ResNetRegressor(random_state=random_state, **kwargs)
    elif (
        r == "individualinceptionregressor"
        or r == "singleinception"
        or r == "individualinception"
    ):
        from aeon.regression.deep_learning import IndividualInceptionRegressor

        return IndividualInceptionRegressor(random_state=random_state, **kwargs)
    elif r == "inceptiontimeregressor" or r == "inception" or r == "inceptiontime":
        from aeon.regression.deep_learning import InceptionTimeRegressor

        return InceptionTimeRegressor(random_state=random_state, **kwargs)

    elif r == "h-inceptiontimeregressor" or r == "h-inceptiontime":
        from aeon.regression.deep_learning import InceptionTimeRegressor

        return InceptionTimeRegressor(
            use_custom_filters=True, random_state=random_state, **kwargs
        )
    elif r == "litetimeregressor" or r == "litetime":
        from aeon.regression.deep_learning import LITETimeRegressor

        return LITETimeRegressor(random_state=random_state, **kwargs)
    elif r == "individualliteregressor" or r == "individuallite":
        from aeon.regression.deep_learning import IndividualLITERegressor

        return IndividualLITERegressor(random_state=random_state, **kwargs)
    elif r == "disjointcnnregressor" or r == "disjointcnn":
        from aeon.regression.deep_learning import DisjointCNNRegressor

        return DisjointCNNRegressor(random_state=random_state, **kwargs)


def _set_regressor_distance_based(
    r, random_state, n_jobs, fit_contract, checkpoint, kwargs
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
    r, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if r == "summary-500":
        from aeon.regression.feature_based import SummaryRegressor
        from sklearn.ensemble import RandomForestRegressor

        return SummaryRegressor(
            estimator=RandomForestRegressor(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "summaryregressor" or r == "summary":
        from aeon.regression.feature_based import SummaryRegressor

        return SummaryRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif r == "catch22-500":
        from aeon.regression.feature_based import Catch22Regressor
        from sklearn.ensemble import RandomForestRegressor

        return Catch22Regressor(
            estimator=RandomForestRegressor(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "catch22regressor" or r == "catch22":
        from aeon.regression.feature_based import Catch22Regressor

        return Catch22Regressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif r == "freshprinceregressor" or r == "fresh-prince" or r == "freshprince":
        from aeon.regression.feature_based import FreshPRINCERegressor

        return FreshPRINCERegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "freshprince-500":
        from aeon.regression.feature_based import FreshPRINCERegressor

        return FreshPRINCERegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "tsfresh-nofs":
        from aeon.regression.feature_based import TSFreshRegressor

        return TSFreshRegressor(
            relevant_feature_extractor=False,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "tsfreshregressor" or r == "tsfresh":
        from aeon.regression.feature_based import TSFreshRegressor

        return TSFreshRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif r == "fpcaregressor" or r == "fpcregressor" or r == "fpcr":
        from tsml.feature_based import FPCARegressor

        return FPCARegressor(n_jobs=n_jobs, **kwargs)
    elif r == "fpcar-b-spline" or r == "fpcr-b-spline" or r == "fpcr-bs":
        from tsml.feature_based import FPCARegressor

        return FPCARegressor(n_jobs=n_jobs, bspline=True, order=4, n_basis=10, **kwargs)


def _set_regressor_hybrid(r, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if r == "ristregressor" or r == "rist" or r == "rist-extrat":
        from aeon.regression.hybrid import RISTRegressor
        from sklearn.ensemble import ExtraTreesRegressor

        return RISTRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=ExtraTreesRegressor(n_estimators=500),
            **kwargs,
        )


def _set_regressor_interval_based(
    r, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if r == "timeseriesforestregressor" or r == "tsf":
        from aeon.regression.interval_based import TimeSeriesForestRegressor

        return TimeSeriesForestRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
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
            **kwargs,
        )
    elif r == "rise-500":
        from aeon.regression.interval_based import (
            RandomIntervalSpectralEnsembleRegressor,
        )

        return RandomIntervalSpectralEnsembleRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "randomintervalspectralensembleregressor" or r == "rise":
        from aeon.regression.interval_based import (
            RandomIntervalSpectralEnsembleRegressor,
        )

        return RandomIntervalSpectralEnsembleRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "cif-500":
        from aeon.regression.interval_based import CanonicalIntervalForestRegressor

        return CanonicalIntervalForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "canonicalintervalforestregressor" or r == "cif":
        from aeon.regression.interval_based import CanonicalIntervalForestRegressor

        return CanonicalIntervalForestRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "drcif" or r == "drcifregressor":
        from aeon.regression.interval_based import DrCIFRegressor

        return DrCIFRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif r == "drcif-500":
        from aeon.regression.interval_based import DrCIFRegressor

        return DrCIFRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            time_limit_in_minutes=fit_contract,
            **kwargs,
        )
    elif r == "summary-intervals":
        from aeon.regression.interval_based import RandomIntervalRegressor
        from aeon.transformations.collection.feature_based import SevenNumberSummary
        from sklearn.ensemble import RandomForestRegressor

        return RandomIntervalRegressor(
            features=SevenNumberSummary(),
            estimator=RandomForestRegressor(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif r == "randomintervals-500" or r == "catch22-intervals-500":
        from aeon.regression.interval_based import RandomIntervalRegressor
        from sklearn.ensemble import RandomForestRegressor

        return RandomIntervalRegressor(
            estimator=RandomForestRegressor(n_estimators=500),
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif (
        r == "randomintervalregressor"
        or r == "randomintervals"
        or r == "catch22-intervals"
    ):
        from aeon.regression.interval_based import RandomIntervalRegressor

        return RandomIntervalRegressor(
            random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif r == "quantregressor" or r == "quant":
        from aeon.regression.interval_based import QUANTRegressor

        return QUANTRegressor(random_state=random_state, **kwargs)


def _set_regressor_other(r, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if r == "dummy" or r == "dummyregressor" or r == "dummyregressor-aeon":
        from aeon.regression import DummyRegressor

        return DummyRegressor(**kwargs)
    elif r == "dummyregressor-tsml":
        from tsml.dummy import DummyRegressor

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
    r, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if r == "rdstregressor" or r == "rdst":
        from aeon.regression.shapelet_based import RDSTRegressor

        return RDSTRegressor(random_state=random_state, n_jobs=n_jobs, **kwargs)


def _set_regressor_vector(r, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if r == "rotationforestregressor" or r == "rotf" or r == "rotationforest":
        from aeon.regression.sklearn import RotationForestRegressor

        return RotationForestRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
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
