"""Hacky area to test shit out"""
import time
from sktime.datasets import load_from_tsfile
from sktime.utils.sampling import stratified_resample
from sktime.distances import dtw_distance
import numpy as np
instance1 = np.array([[1,2,3,4], [4,3,2,1]])
instance2 = np.array([[2,3,4,5], [5,4,3,2]])
print(" shape is [n_dimensions, series_length] = ", instance1.shape)
print(" DTW_D is = ", dtw_distance(instance1, instance2))


def time_data_load():
    dataset = ["InsectWingbeatEq"]
    for file in dataset:
        start = time.time()
        x, y = load_from_tsfile(f"C:/Data/{file}/{file}_TRAIN.ts")
        x2, y2 = load_from_tsfile(f"C:/Data/{file}/{file}_TEST.ts")
        end = time.time()
        print(f" Load pandas for problem {file} time taken = {end-start}")
        start = time.time()
        x, y, x2, y2 = stratified_resample(x, y, x2, y2, 1)
        end = time.time()
        print(f" resample time problem  {file} time taken = {end-start}")
#        start = time.time()
#        x, y = load_from_tsfile(f"C:/Data/{file}/{file}_TRAIN.ts",
#                                return_data_type="numpy3d")
#        x2, y2 = load_from_tsfile(f"C:/Data/{file}/{file}_TEST.ts",
#                                return_data_type="numpy3d")
#        end = time.time()
#        print(f" Load numpy for problem  {file} time taken = {end-start}")


time_data_load()