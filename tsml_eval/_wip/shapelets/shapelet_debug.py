"""File to debug changes in STC and STC."""

# Changes to the core ST algorithm from aeon.

from tsml_eval._wip.shapelets.classification.shapelet_based import ShapeletTransformClassifier \
    as WIP_STC
from aeon.classification.shapelet_based import ShapeletTransformClassifier as STC
from aeon.datasets import load_unit_test


def compare_outputs():
    trainX, trainY= load_unit_test(split="TRAIN")
    testX, testY = load_unit_test(split="TEST")
    stc = STC(random_state=0)
    wip_stc = WIP_STC(random_state=0)
    wip2_stc = WIP_STC(random_state=0, shapelet_quality="F_STAT")
    stc.fit(trainX, trainY)
    wip_stc.fit(trainX, trainY)
    wip2_stc.fit(trainX, trainY)
    st_preds = stc.predict(testX)
    wip_st_preds = wip_stc.predict(testX)
    wip2_st_preds = wip2_stc.predict(testX)
    print(st_preds)
    print(wip_st_preds)
    print(wip2_st_preds)


if __name__ == "__main__":
    compare_outputs()
