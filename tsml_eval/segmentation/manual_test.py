from _sw import SlidingWindow
from _bu import BottomUp
from _td import TopDown
from _swab import SWAB
from aeon.datasets import load_electric_devices_segmentation
from aeon.visualisation import plot_series_with_change_points, plot_series_with_profiles
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


ts, period_size, true_cps = load_electric_devices_segmentation()
ts = ts.values
ts = ts.reshape((len(ts), 1))
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(ts)
ts = scaler.transform(ts)
pla = BottomUp(22)
results = pla.dense(ts)
print(results)
print(true_cps)