"""Work-in-progress regression estimators."""

from tsml_eval._wip.regression._convtran import ConvTranRegressor
from tsml_eval._wip.regression._timesnet import TimesNetRegressor
from tsml_eval._wip.regression._ts2vec import TS2VecRegressor

__all__ = ["ConvTranRegressor", "TS2VecRegressor", "TimesNetRegressor"]
