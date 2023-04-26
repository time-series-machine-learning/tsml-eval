# -*- coding: utf-8 -*-

from aeon.distances import dtw_distance, dtw_alignment_path
from aeon.distances import distance_factory, distance, distance_alignment_path
from aeon.distances import pairwise_distance
import numpy as np

a = np.array(
    [0.018, 1.537, -0.141, -0.761, -0.177, -2.192, -0.193, -0.465, -0.944, -0.240]
)
b = np.array([-0.755, 0.446, 1.198, 0.171, 0.564, 0.689, 1.794, 0.066, 0.288, 1.634])
from aeon.clustering.k_means import TimeSeriesKMeans
from aeon.clustering.k_medoids import TimeSeriesKMedoids
from aeon.datasets import load_arrow_head, load_unit_test
from aeon.benchmarking.experiments import run_clustering_experiment

## Code listing 1
d1 = dtw_distance(a, b)
d2 = dtw_distance(a, b, window=0.2)
d3 = distance(a, b, metric="dtw", window=0.2)
p1 = dtw_alignment_path(a, b)
p2 = dtw_alignment_path(a, b, window=0.2)
p3 = distance_alignment_path(a, b, metric="dtw")
print("Full window DTW distance  = ", d1)
print("Full window path  = ", p1)
# For repeated use create a numba compiled callable for performance
dtw_numba = distance_factory(a, b, metric="dtw")
# d3 = dtw_numba(a, b)
# dtw_numba = distance_factory(a, b, metric='dtw')
# msm_numba = distance_factory(a, b, metric="msm")
# d3 = dtw_numba(a, b)
# msm_numba = distance_factory(a, b, metric="msm")
# d4 = msm_numba(a, b)
# print("Full window DTW distance  = ",d1)

# pair = np.array([a, b])
# dist = pairwise_distance(pair, pair, metric="twe")
# print(dist)

## Code listing 2
trainX, trainY = load_unit_test(split="test", return_type="numpy2D")
testX, testY = load_unit_test(split="train", return_type="numpy2D")
clst1 = TimeSeriesKMeans()
clst2 = TimeSeriesKMeans(
    averaging_method="dba",
    metric="dtw",
    distance_params={"window": 0.1},
    n_clusters=2,
    random_state=1,
)
clst3 = TimeSeriesKMedoids()
clst4 = TimeSeriesKMedoids(
    metric="dtw",
    distance_params={"window": 0.2},
    n_clusters=len(set(trainY)),
    random_state=1,
)
run_clustering_experiment(
    trainX,
    clst1,
    results_path="temp/",
    trainY=trainY,
    testX=testX,
    testY=testY,
    cls_name="kmeans",
    dataset_name="UnitTest",
    resample_id=0,
    overwrite=False,
)


clst1.fit(trainX)
pred = clst1.predict(testX)
print(pred)
clst2.fit(trainX)
pred = clst2.predict(testX)
print(pred)
clst3.fit(trainX)
pred = clst3.predict(testX)
print(pred)
clst4.fit(trainX)
pred = clst4.predict(testX)
print(pred)
