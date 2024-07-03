from _pla import PiecewiseLinearApproximation
from aeon.datasets import load_electric_devices_segmentation
from aeon.visualisation import plot_series_with_change_points, plot_series_with_profiles
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


ts, period_size, true_cps = load_electric_devices_segmentation()
ts = ts[:30]
ts = ts.values


ts = np.array([573.0,375.0,301.0,212.0,55.0,34.0,25.0,33.0,113.0,143.0,303.0,615.0,1226.0,1281.0,1221.0,1081.0,866.0,1096.0,1039.0,975.0,746.0,581.0,409.0,182.0])

pla = PiecewiseLinearApproximation(PiecewiseLinearApproximation.Transformer.BottomUp, 5)
results = pla.fit_transform(ts)

print("Original: ", ts)
print("PLA     : ", results)

plt.subplot(2, 1, 1)  # (rows, columns, subplot_number)
plt.plot(np.arange(len(ts)), ts)
plt.title('Original')
plt.xlabel('x')
plt.ylabel('y1')

# Create the second subplot (lower plot)
plt.subplot(2, 1, 2)  # (rows, columns, subplot_number)
plt.plot(np.arange(len(ts)), results)
plt.title('PLA')
plt.xlabel('x')
plt.ylabel('y2')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Display the plot
plt.show()