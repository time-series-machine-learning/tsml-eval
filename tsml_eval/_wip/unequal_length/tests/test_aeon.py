from aeon.testing.estimator_checking import parametrize_with_checks

from tsml_eval._wip.unequal_length._arsenal import Arsenal
from tsml_eval._wip.unequal_length._pad import Padder
from tsml_eval._wip.unequal_length._rocket import RocketClassifier
from tsml_eval._wip.unequal_length._stc import ShapeletTransformClassifier


@parametrize_with_checks([Arsenal, ShapeletTransformClassifier, RocketClassifier, Padder])
def test_aeon_estimator(check):
    check()
