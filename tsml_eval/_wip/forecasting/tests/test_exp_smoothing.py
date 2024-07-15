"""Tests for the exponential smoothing conversion functions."""

import pytest
import pandas as pd
import numpy as np

from tsml_eval._wip.forecasting.exponential_smoothing import fit_ets, MAX_NMSE
from aeon.datasets import load_airline


expected_results = {
    'lik': 2073.6048878951306,
    'error':    [0., 6., 19.89942, 16.5658394, 8.28814072, 22.14920385, 34.77790937, 34.1949157, 21.62169494, 4.25924327,
                -10.81215585, 3.36909192, 0.31261471, 11.30737425, 26.11782497, 19.68000317, 9.35010072, 33.19336187, 53.63693049, 52.73779677,
                39.85373552, 14.18565408, -5.05214477, 21.03254602, 25.67997045, 30.24948855, 57.7424063, 41.77445112, 50.07417208, 55.23476206,
                75.30884335, 74.04641613, 57.8051514, 34.8361444, 18.25217451, 37.94620723, 42.31010232, 50.60084399, 62.75260519, 49.70066237,
                50.86751362, 85.01480455, 95.58967307, 105.98727155, 71.21057162, 52.0168451, 32.1448694, 53.60601425, 54.70739878, 53.79032044,
                92.88861539, 90.3314926, 82.8172357, 95.4289428, 114.82923566, 120.90431494, 83.87755565, 56.47148825, 24.52483789, 45.11371987,
                47.35746356, 30.56359463, 77.05124691, 67.7596112, 73.62373427, 102.3895551, 138.67316489, 127.34854045, 91.21375446, 59.68470792,
                32.68419329, 58.13629727, 70.16173916, 59.98559456, 92.98003607, 93.42138076, 92.85532704, 136.29876228, 183.01394074, 162.9460171,
                125.21449876, 85.11548642, 46.68866718, 86.9060095, 91.44917512, 82.91618215, 121.52623058, 115.48904591, 118.55306457, 172.56572007,
                208.67294344, 197.1748894, 143.86958107, 92.45784738, 55.90794569, 89.97074218, 97.46253267, 81.82873578, 135.45701343, 125.18630241,
                130.08776274, 194.90705825, 234.63976633, 232.70642179, 165.80548655, 106.02603397, 62.24868425, 92.20518883, 94.65952254, 71.07271345,
                113.88129788, 97.97226776, 111.32992602, 181.46366539, 234.42172954, 244.49204003, 139.39353854, 92.05683824, 41.51365881, 67.81775152,
                89.68089997, 70.17754918, 133.00113955, 120.77159716, 142.74706266, 192.35414611, 265.12964951, 271.68519291, 171.13084355, 112.26212023,
                65.38023292, 107.28424231, 117.48580083, 89.51634723, 116.01575489, 156.07094416, 164.45467495, 224.69786647, 307.93118131, 286.76922837,
                183.96202031, 133.8782037, 60.63395879, 101.61753155],
    'amse':     [12459.27549448, 12950.04315057, 13422.75326373,     0.,
                0.,             0.,             0.,             0.,
                0.,             0.,             0.,             0.,
                0.,             0.,             0.,             0.,
                0.,             0.,             0.,             0.,
                0.,             0.,             0.,             0.,
                0.,             0.,             0.,             0.,
                0.,             0.,        ]
}

y = load_airline().to_numpy()
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

@pytest.mark.parametrize("error, trend, season", [
    (0, 0, 0),  # no error, additive trend, no seasonality (simple exponential smoothing)
    (1, 1, 0),  # additive error, additive trend, no seasonality (Holt's linear method)
    (2, 1, 1),  # multiplicative error, additive trend, additive seasonality (additive Holt-Winters' method)
    (1, 1, 2)   # additive error, additive trend, multiplicative seasonality (multiplicative Holt-Winters' method)
])
def test_fit_etsc(error, trend, season):
    """Test the etsc function."""
    l0 = y[0]
    b0 = (y[1] - y[0]) / 2
    s0 = [y[i] - l0 for i in range(m)]
    x = np.array([l0, b0] + s0, dtype=float)
    nstates = 1 + (trend > 0) + m * (season > 0)  # expanding x to hold the states for each time step
    x = np.concatenate([x, np.zeros(n * nstates)]).flatten()  # fit_ets expects 'x' to be a one dimensional array

    fit_ets(y, n, x, m, error, trend, season, alpha, beta, gamma, phi, e, lik, amse, nmse)

    assert lik[0] != 0
    assert np.all(amse[:nmse] >= 0)
    assert np.trim_zeros(amse).size == nmse

    return

def test_fit_etsc_output():
    """Test the correctness of output compared to Hyndman's etscalc function"""
    error, trend, season = 1, 1, 0
    l0 = y[0]
    b0 = (y[1] - y[0]) / 2
    s0 = [y[i] - l0 for i in range(m)]
    x = np.array([l0, b0] + s0, dtype=float)
    nstates = 1 + (trend > 0) + m * (season > 0)  # expanding x to hold the states for each time step
    x = np.concatenate([x, np.zeros(n * nstates)]).flatten()  # fit_ets expects 'x' to be a one dimensional array

    fit_ets(y, n, x, m, error, trend, season, alpha, beta, gamma, phi, e, lik, amse, nmse)

    assert np.isclose(lik, expected_results['lik'])
    assert np.allclose(e, expected_results['error'])
    assert np.allclose(amse, expected_results['amse'])

    return
