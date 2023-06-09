# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
from aeon.clustering.metrics.averaging import elastic_barycenter_average
from aeon.datasets import load_from_tsfile

dataset = "GunPoint"
c = "1"
plt.figure()

fig, axs = plt.subplots(3, 2, sharey=True, sharex=True, figsize=(8, 6))

x_train, y_train = load_from_tsfile(
    os.path.join(f"../../../../../ajb/Data/{dataset}/{dataset}_TRAIN.ts")
)

_, _, len_ts = x_train.shape

x = range(0, len_ts)

idxs = np.where(y_train == c)

for i in x_train[idxs]:
    axs[0, 0].plot(x, i[0], lw=0.2)
    axs[1, 0].plot(x, i[0], lw=0.2)
    axs[2, 0].plot(x, i[0], lw=0.2)


series_avg = np.mean(np.array(x_train[idxs]), axis=0)[0]

axs[0, 1].plot(x, series_avg, color="red")

series_mba = elastic_barycenter_average(
    x_train[idxs],
    metric="msm",
)


axs[1, 1].plot(x, series_mba[0, :])

series_dba = elastic_barycenter_average(
    x_train[idxs],
    metric="dtw",
)

axs[2, 1].plot(x, series_dba[0, :], color="green")

fig.suptitle(f"{dataset} - Class {c}")
axs[0, 0].set_title("Original time series")
axs[0, 1].set_title("Averaging")
axs[1, 0].set_title("Original time series")
axs[1, 1].set_title("MBA")
axs[2, 0].set_title("Original time series")
axs[2, 1].set_title("DBA")
fig.tight_layout()

plt.savefig("test.png")
