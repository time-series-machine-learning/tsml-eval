"""WIP classification estimators, starting with DrCIF speedup experiments."""

__all__ = [
    "NewDrCIF",
    "QuantDrCIF",
    "HC2Quant",
    "BaggedQUANT",
    "SharedDrCIF",
    "FastDrCIF",
    "FastDrCIF_D",
    "RDSTDrCIF",
    "PULSARClassifier",
    "FIRE",
]

from tsml_eval._wip.classification._bagged_quant import BaggedQUANT
from tsml_eval._wip.classification._fire import FIRE
from tsml_eval._wip.classification._pulsar import PULSARClassifier
from tsml_eval._wip.classification._fast_drcif import (
    FastDrCIF,
    FastDrCIF_D,
    RDSTDrCIF,
)
from tsml_eval._wip.classification._hc2_quant import HC2Quant
from tsml_eval._wip.classification._new_drcif import NewDrCIF
from tsml_eval._wip.classification._quant_drcif import QuantDrCIF
from tsml_eval._wip.classification._shared_drcif import SharedDrCIF
