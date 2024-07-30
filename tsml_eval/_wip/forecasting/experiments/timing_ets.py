import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsml_eval._wip.forecasting.exponential_smoothing import fit_ets, MAX_NMSE

fit_ets_standard = fit_ets.py_func  # function without numba decorator
input_sizes = [10, 100, 1000, 10000, 100000, 1000000]

results = []

for i, n in enumerate(input_sizes):
    y = np.random.rand(n)
    n = len(y)
    m = 4
    alpha = 0.016763333
    beta = 0.001766333
    gamma = 0.
    phi = 0.
    e = np.zeros(n)
    lik = np.zeros(1)
    amse = np.zeros(MAX_NMSE)
    nmse = 3
    error, trend, season = 1, 1, 0
    l0 = y[0]
    b0 = (y[1] - y[0]) / 2
    s0 = [y[i] - l0 for i in range(m)]
    x = np.array([l0, b0] + s0, dtype=float)
    nstates = 1 + (trend > 0) + m * (season > 0)
    x = np.concatenate([x, np.zeros(n * nstates)]).flatten()

    start_time = time.time()
    fit_ets_standard(y, n, x, m, error, trend, season, alpha, beta, gamma, phi, e, lik, amse, nmse)
    end_time = time.time()
    t_standard = end_time - start_time

    start_time = time.time()
    fit_ets(y, n, x, m, error, trend, season, alpha, beta, gamma, phi, e, lik, amse, nmse)
    end_time = time.time()
    t_numba = end_time - start_time

    if i >= 1:  # discard first measurement
        results.append((n, t_standard, t_numba))
results_df = pd.DataFrame(results, columns=['Input Size', 'fit_ets', 'fit_ets (numba)'])
results_df.set_index('Input Size', inplace=True)
results_df.to_csv('tsml_eval/_wip/forecasting/experiments/timing_results.csv')

fig, ax = plt.subplots()
ax.plot(results_df.index, results_df['fit_ets'], label='fit_ets')
ax.plot(results_df.index, results_df['fit_ets (numba)'], label='fit_ets (numba)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Input Size')
ax.set_ylabel('Time (s)')
ax.legend()
plt.show()
