"""File to debug changes in STC and STC."""

""""Changes to the core ST algorithm from aeon.


"""

from tsml_eval._wip.classification.shapelet_based import ShapeletTransformClassifier \
    as WIP_STC
from tsml_eval._wip.transformations.collection.shapelet_based import \
    RandomShapeletTransform as WIP_ST
from aeon.classification.shapelet_based import ShapeletTransformClassifier as STC
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform as ST
from aeon.datasets import load_unit_test


def compare_outputs():
    trainX, trainY= load_unit_test(split="TRAIN")
    testX, testY = load_unit_test(split="TEST")
    stc = STC(random_state=0)
    wip_stc = WIP_STC(random_state=0)
    stc.fit(trainX, trainY)
    wip_stc.fit(trainX, trainY)
    st_preds = stc.predict(testX)
    wip_st_preds = wip_stc.predict(testX)
    print(st_preds)
    print(wip_st_preds)



if __name__ == "__main__":
    compare_outputs()

