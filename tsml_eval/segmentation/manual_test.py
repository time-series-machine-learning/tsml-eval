from _sw import SlidingWindow
from _bu import BottomUp
from _td import TopDown
from aeon.datasets import load_electric_devices_segmentation
from aeon.visualisation import plot_series_with_change_points, plot_series_with_profiles
import matplotlib.pyplot as plt
import numpy as np


ts, period_size, true_cps = load_electric_devices_segmentation()
ts = ts.values
sw = SlidingWindow(100)
results = sw.sliding_window(ts)
print(len(results))

plt.figure()
plt.plot(np.arange(len(ts)), ts)
plt.title('original')
plt.xlabel('x')
plt.ylabel('y')

flattened_arr = [item for sublist in results for item in sublist]
plt.figure()
plt.plot(np.arange(len(flattened_arr)), flattened_arr)
plt.title('pla')
plt.xlabel('x')
plt.ylabel('y')

for i in range(len(results)):
    plt.figure()
    plt.plot(np.arange(len(results[i])), results[i])
    plt.title(i)
    plt.xlabel('x')
    plt.ylabel('y')

plt.show()
