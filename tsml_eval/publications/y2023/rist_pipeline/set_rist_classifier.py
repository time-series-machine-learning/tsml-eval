"""Classifiers used in the publication."""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np

from tsml_eval.utils.functions import str_in_nested_list

rist_classifiers = [
    ["FreshPRINCEClassifier", "freshprince"],
    ["ShapeletTransformClassifier", "stc", "stc-2hour"],
    ["RDSTClassifier", "rdst"],
    ["RSTSFClassifier", "rstsf", "r-stsf"],
    ["drcif", "DrCIFClassifier"],
    ["RocketClassifier", "rocket"],
    ["HIVECOTEV2", "hc2"],
    ["RISTClassifier", "rist", "rist-extrat"],
    "rist-rf",
    "rist-ridgecv",
    ["RandomIntervalClassifier", "intervalpipeline", "i-pipeline"],
]


def _set_rist_classifier(
    classifier_name,
    random_state=None,
    n_jobs=1,
    **kwargs,
):
    c = classifier_name.lower()

    if not str_in_nested_list(rist_classifiers, c):
        raise ValueError(f"UNKNOWN CLASSIFIER: {c} in _set_rist_classifier")

    if c == "freshprinceclassifier" or c == "freshprince":
        from aeon.classification.feature_based import FreshPRINCEClassifier

        return FreshPRINCEClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "shapelettransformclassifier" or c == "stc" or c == "stc-2hour":
        from aeon.classification.shapelet_based import ShapeletTransformClassifier

        return ShapeletTransformClassifier(
            transform_limit_in_minutes=120,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "rdstclassifier" or c == "rdst":
        from aeon.classification.shapelet_based import RDSTClassifier

        return RDSTClassifier(random_state=random_state)
    elif c == c == "rstsfclassifier" or c == "rstsf" or c == "r-stsf":
        from aeon.classification.interval_based import RSTSF

        return RSTSF(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
    elif c == "drcif" or c == "drcifclassifier":
        from aeon.classification.interval_based import DrCIFClassifier

        return DrCIFClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif c == "rocketclassifier" or c == "rocket":
        from aeon.classification.convolution_based import RocketClassifier

        return RocketClassifier(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "hivecotev2" or c == "hc2":
        from aeon.classification.hybrid import HIVECOTEV2

        return HIVECOTEV2(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "ristclassifier" or c == "rist" or c == "rist-extrat":
        from aeon.classification.hybrid import RISTClassifier
        from sklearn.ensemble import ExtraTreesClassifier

        return RISTClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=ExtraTreesClassifier(n_estimators=500, criterion="entropy"),
            **kwargs,
        )
    elif c == "rist-rf":
        from aeon.classification.hybrid import RISTClassifier
        from sklearn.ensemble import RandomForestClassifier

        return RISTClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=RandomForestClassifier(n_estimators=500, criterion="entropy"),
            **kwargs,
        )
    elif c == "rist-ridgecv":
        from aeon.classification.hybrid import RISTClassifier
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return RISTClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            estimator=make_pipeline(
                StandardScaler(with_mean=False),
                RidgeClassifierCV(alphas=np.logspace(-4, 4, 20)),
            ),
            **kwargs,
        )
    elif (
        c == "randomintervalclassifier" or c == "intervalpipeline" or c == "i-pipeline"
    ):
        from aeon.transformations.collection.feature_based import Catch22
        from sklearn.ensemble import ExtraTreesClassifier
        from tsml.interval_based import RandomIntervalClassifier
        from tsml.transformations import (
            ARCoefficientTransformer,
            FunctionTransformer,
            PeriodogramTransformer,
        )
        from tsml.utils.numba_functions.general import first_order_differences_3d
        from tsml.utils.numba_functions.stats import (
            row_iqr,
            row_mean,
            row_median,
            row_numba_max,
            row_numba_min,
            row_ppv,
            row_slope,
            row_std,
        )

        def sqrt_times_15_plus_5_mv(X):
            return int(
                np.sqrt(X.shape[2]) * np.sqrt(X.shape[1]) * 15 + 5
            )  # pragma: no cover

        interval_features = [
            Catch22(outlier_norm=True, replace_nans=True),
            row_mean,
            row_std,
            row_slope,
            row_median,
            row_iqr,
            row_numba_min,
            row_numba_max,
            row_ppv,
        ]
        series_transformers = [
            None,
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(use_pyfftw=True),
            ARCoefficientTransformer(replace_nan=True),
        ]

        return RandomIntervalClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            n_intervals=sqrt_times_15_plus_5_mv,
            features=interval_features,
            series_transformers=series_transformers,
            estimator=ExtraTreesClassifier(n_estimators=500, criterion="entropy"),
            **kwargs,
        )
