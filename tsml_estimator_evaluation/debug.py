"""Hacky area to test shit out"""
import time
from sktime.datasets import load_from_tsfile


def time_data_load():
    dataset = ["Tiselac"]
    for x in dataset:
        start = time.time()
        x, y = load_from_tsfile("C:/Data/Tiselac/Tiselac_TRAIN.ts")
        x, y = load_from_tsfile("C:/Data/Tiselac/Tiselac_TEST.ts")
        end = time.time()
        print(f" Load pandas for problem {x} time taken = {end-start}")
        start = time.time()
        x, y = load_from_tsfile("C:/Data/Tiselac/Tiselac_TRAIN.ts",
                                return_data_type="numpy3d")
        x, y = load_from_tsfile("C:/Data/Tiselac/Tiselac_TEST.ts",
                                return_data_type="numpy3d")
        end = time.time()
        print(f" Load numpy for problem  {x} time taken = {end-start}")


time_data_load()