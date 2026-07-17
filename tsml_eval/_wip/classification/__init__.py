"""WIP classification estimators, starting with DrCIF speedup experiments."""

__all__ = [
    "NewDrCIF",
    "QuantDrCIF",
    "HC2Quant",
    "BaggedQUANT",
    "SharedDrCIF",
    "SharedDrCIF2",
]

from tsml_eval._wip.classification._bagged_quant import BaggedQUANT
from tsml_eval._wip.classification._hc2_quant import HC2Quant
from tsml_eval._wip.classification._new_drcif import NewDrCIF
from tsml_eval._wip.classification._quant_drcif import QuantDrCIF
from tsml_eval._wip.classification._shared_drcif import SharedDrCIF
from tsml_eval._wip.classification._shared_drcif2 import SharedDrCIF2
