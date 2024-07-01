from _sw import SlidingWindow
from _bu import BottomUp
from _td import TopDown
from _swab import SWAB
from aeon.datasets import load_electric_devices_segmentation
from aeon.visualisation import plot_series_with_change_points, plot_series_with_profiles
import matplotlib.pyplot as plt
import numpy as np


ts, period_size, true_cps = load_electric_devices_segmentation()
ts = ts.values
sw = SWAB(100)
results = sw.swab(ts)
print(len(results))

print(results)