# -*- coding: utf-8 -*-
import pytest
from sktime.utils.estimator_checks import check_estimator

from tsml_estimator_evaluation.sktime_estimators.classification.hydra import HYDRA
from tsml_estimator_evaluation.sktime_estimators.classification.muse_dilation import (
    MUSE_DILATION,
)
from tsml_estimator_evaluation.sktime_estimators.classification.rdst import (
    RDST,
    RDSTEnsemble,
)
from tsml_estimator_evaluation.sktime_estimators.classification.weasel_dilation import (
    WEASEL_DILATION,
)
from tsml_estimator_evaluation.sktime_estimators.transformations.sfa_dilation import (
    SFADilation,
)


@pytest.mark.parametrize(
    "est", [SFADilation, WEASEL_DILATION, MUSE_DILATION, HYDRA, RDST, RDSTEnsemble]
)
def test_check_estimator(est):
    check_estimator(est, return_exceptions=False)
