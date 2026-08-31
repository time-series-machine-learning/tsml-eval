"""Work-in-progress classification estimators."""

from tsml_eval._wip.classification._convtran import ConvTranClassifier
from tsml_eval._wip.classification._patchmtsc import PatchMTSCClassifier
from tsml_eval._wip.classification._timesurl import TimesURLClassifier
from tsml_eval._wip.classification._timesnet import TimesNetClassifier
from tsml_eval._wip.classification._ts2vec import TS2VecClassifier

__all__ = ["ConvTranClassifier", "PatchMTSCClassifier", "TimesURLClassifier", "TimesNetClassifier", "TS2VecClassifier"]
