"""Work-in-progress classification estimators."""

from tsml_eval._wip.classification._convtran import ConvTranClassifier
from tsml_eval._wip.classification._patchmtsc import PatchMTSCClassifier
from tsml_eval._wip.classification._timesurl import TimesURLClassifier

__all__ = ["ConvTranClassifier", "PatchMTSCClassifier", "TimesURLClassifier"]
