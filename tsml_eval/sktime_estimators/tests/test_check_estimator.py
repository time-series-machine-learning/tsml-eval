# -*- coding: utf-8 -*-
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks
from sktime.utils.estimator_checks import check_estimator

from tsml_eval.sktime_estimators.classification.hydra import HYDRA
from tsml_eval.sktime_estimators.classification.mpdist import MPDist
from tsml_eval.sktime_estimators.classification.muse_dilation import MUSE_DILATION
from tsml_eval.sktime_estimators.classification.rdst import RDST, RDSTEnsemble
from tsml_eval.sktime_estimators.classification.rsf import RandomShapeletForest
from tsml_eval.sktime_estimators.classification.weasel_dilation import WEASEL_DILATION
from tsml_eval.sktime_estimators.regression.convolution_based.arsenal import Arsenal
from tsml_eval.sktime_estimators.regression.dictionary_based.tde import (
    TemporalDictionaryEnsemble,
)
from tsml_eval.sktime_estimators.regression.hybrid.hivecote_v2 import HIVECOTEV2
from tsml_eval.sktime_estimators.regression.interval_based.drcif import DrCIF
from tsml_eval.sktime_estimators.regression.shapelet_based.str import (
    ShapeletTransformRegressor,
)
from tsml_eval.sktime_estimators.regression.sklearn.rotation_forest import (
    RotationForest,
)
from tsml_eval.sktime_estimators.regression.transformations.sfa import SFA
from tsml_eval.sktime_estimators.regression.transformations.shapelet_transform import (
    RandomShapeletTransform,
)
from tsml_eval.sktime_estimators.transformations.sfa_dilation import SFADilation

classification_estimators = [
    # WEASEL_DILATION,
    # MUSE_DILATION,
    SFADilation,
    HYDRA,
    RDST,
    RDSTEnsemble,
    RandomShapeletForest,
    MPDist,
]
regression_estimators = [
    DrCIF,
    ShapeletTransformRegressor,
    # RandomShapeletTransform,
    Arsenal,
    TemporalDictionaryEnsemble,
    # SFA,
    HIVECOTEV2,
]


@pytest.mark.parametrize("est", classification_estimators + regression_estimators)
def test_check_estimator(est):
    check_estimator(est, return_exceptions=False)


regression_sklearn_estimators = [RotationForest(n_estimators=3)]


@parametrize_with_checks(regression_sklearn_estimators)
def test_sklearn_compatible_estimator(estimator, check):
    """Test that sklearn estimators adhere to sklearn conventions."""
    check(estimator)
