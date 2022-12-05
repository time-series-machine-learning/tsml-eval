"""Hacky area to test shit out"""
import time
from sktime.datasets import load_from_tsfile


def time_data_load():
    dataset = ["FaceDetection"]
    for file in dataset:
        start = time.time()
        x, y = load_from_tsfile(f"C:/Data/{file}/{file}_TRAIN.ts")
        x, y = load_from_tsfile(f"C:/Data/{file}/{file}_TEST.ts")
        end = time.time()
        print(f" Load pandas for problem {file} time taken = {end-start}")
        start = time.time()
        x, y = load_from_tsfile(f"C:/Data/{file}/{file}_TRAIN.ts",
                                return_data_type="numpy3d")
        x, y = load_from_tsfile(f"C:/Data/{file}/{file}_TEST.ts",
                                return_data_type="numpy3d")
        end = time.time()
        print(f" Load numpy for problem  {file} time taken = {end-start}")


time_data_load()