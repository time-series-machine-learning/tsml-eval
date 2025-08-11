from aeon.testing.estimator_checking import parametrize_with_checks
from tsml_eval._wip.shapelets.early_abandon._shapelet_transform3 import RandomShapeletTransform as RandomShapeletTransform3

@parametrize_with_checks([RandomShapeletTransform3])
def test_aeon_transform(check):
    check()
