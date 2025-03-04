from aeon.testing.estimator_checking import parametrize_with_checks

from tsml_eval._wip.unequal_length._arsenal import Arsenal
from tsml_eval._wip.unequal_length._drcif import DrCIFClassifier
from tsml_eval._wip.unequal_length._pad import Padder
from tsml_eval._wip.unequal_length._rocket import RocketClassifier
from tsml_eval._wip.unequal_length._stc import ShapeletTransformClassifier
from tsml_eval._wip.unequal_length._tde import TemporalDictionaryEnsemble, IndividualTDE


@parametrize_with_checks([Arsenal, ShapeletTransformClassifier, RocketClassifier, TemporalDictionaryEnsemble, IndividualTDE, DrCIFClassifier, Padder])
def test_aeon_estimator(check):
    check()
