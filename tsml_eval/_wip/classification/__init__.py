"""Work-in-progress classification estimators."""

from tsml_eval._wip.classification._convtran import ConvTranClassifier
from tsml_eval._wip.classification._disjoint_cnn import DisjointCNNClassifier
from tsml_eval._wip.classification._patchmtsc import PatchMTSCClassifier
from tsml_eval._wip.classification._timesnet import TimesNetClassifier
from tsml_eval._wip.classification._timesurl import TimesURLClassifier
from tsml_eval._wip.classification._ts2vec import TS2VecClassifier
from tsml_eval._wip.classification._xcm import XCMClassifier

__all__ = [
    "ConvTranClassifier",
    "DisjointCNNClassifier",
    "PatchMTSCClassifier",
    "TimesNetClassifier",
    "TimesURLClassifier",
    "TS2VecClassifier",
    "XCMClassifier",
]
