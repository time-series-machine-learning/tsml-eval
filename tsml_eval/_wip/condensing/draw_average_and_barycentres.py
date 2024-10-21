import os

import matplotlib.pyplot as plt
import numpy as np
from aeon.clustering.metrics.averaging import elastic_barycenter_average
from aeon.datasets import load_from_tsfile

dataset = "GunPoint"
c = "1"

distances = ["msm", "dtw", "twe"]
distance_params = {
    "msm": {"c": 1},
    "dtw": {"window": 0.2},
    "twe": {"nu": 0.05, "lmbda": 1},
}
names = ["MBA", "DBA", "TBA"]
colours = ["blue", "purple", "green"]
n_methods = len(distances) + 1

fig = plt.figure(figsize=(13, 13))

gs0 = fig.add_gridspec(1, 2)

gs00 = gs0[0].subgridspec(n_methods * 2, 1)
gs01 = gs0[1].subgridspec(n_methods, 1)

# original set of time series
start = n_methods - 1
end = n_methods + 1
ax00_gs00 = fig.add_subplot(gs00[start:end, 0])

x_train, y_train = load_from_tsfile(
    os.path.join(f"../../../../TSC_datasets/{dataset}/{dataset}_TRAIN.ts")
)

x = range(0, x_train.shape[2])
idxs = np.where(y_train == c)

for i in x_train[idxs]:
    ax00_gs00.plot(x, i[0], lw=0.2)

ax00_gs00.set_title("Original time series", size=14)

# average time series
ax01_gs01 = fig.add_subplot(gs01[0])
series_avg = np.mean(np.array(x_train[idxs]), axis=0)[0]
ax01_gs01.plot(x, series_avg, color="red")
ax01_gs01.set_title("Averaging", size=14)

# plots BA time series (msm, dtw, twe).
for idx, i in enumerate(distances):
    series_BA = elastic_barycenter_average(
        x_train[idxs],
        metric=i,
        **distance_params[i],
    )
    ax = fig.add_subplot(gs01[idx + 1])
    ax.plot(x, series_BA[0, :], color=colours[idx])
    ax.set_title(names[idx], size=14)

fig.suptitle(f"{dataset} - Class {c}", size=16)

fig.tight_layout()

plt.savefig("barycentres_example.png")
