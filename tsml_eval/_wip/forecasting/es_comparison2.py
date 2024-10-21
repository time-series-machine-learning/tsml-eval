import time

import numba
import numpy as np
from tsml_eval._wip.forecasting.exponential_smoothing import fit_ets
from statsforecast.ets import etscalc
from statsforecast.utils import AirPassengers as ap

NA = -99999.0
MAX_NMSE = 30
MAX_SEASONAL_PERIOD = 24

def setup(n):
    y = np.random.rand(n)
    m = 4
    error = 1
    trend = 1
    season = 1
    nstates = 1 + (trend > 0) + m * (season > 0)
    init_states = np.zeros(n * (nstates + 1))
    init_states[0] = y[0]
    init_states[1] = (y[1]-y[0]) / 2
    alpha = 0.1
    beta = 0.1
    gamma = 0.1
    phi = 1
    e = np.zeros(n)
    lik_fitets = np.zeros(1)
    amse = np.zeros(MAX_NMSE)
    nmse = 3
    return y, n, init_states, m, error, trend, season, alpha, beta, gamma, phi, e, lik_fitets, amse, nmse


def test_ets_comparison():
    for i in range(1000, 2001, 1000):
        y, n, init_states, m, error, trend, season, alpha, beta, gamma, phi, e, \
            lik_fitets, amse, nmse = setup(i)

        # tsml-eval implementation
        start = time.time()
        fit_ets(y, n, init_states, m,
                error, trend, season,
                alpha, beta, gamma, phi,
                e, lik_fitets, amse, nmse)
        end = time.time()
        time_fitets = end - start

        init_states_fitets = init_states.copy()
        e_fitets = e.copy()
        amse_fitets = amse.copy()

        # Reinitialise arrays
        y, n, init_states, m, error, trend, season, alpha, beta, gamma, phi, e, \
            lik_fitets, amse, nmse = setup(i)

        # Nixtla/statsforcast implementation
        start = time.time()
        # lik_etscalc = etscalc(ap, n, init_states, m,
        #                       error, trend, season,
        #                       alpha, beta, gamma, phi,
        #                       e, amse, nmse)
        end = time.time()
        time_etscalc = end - start

        # init_states_etscalc = init_states.copy()
        # e_etscalc = e.copy()
        # amse_etscalc = amse.copy()

        # Comparing outputs and runtime
        # assert np.allclose(init_states_fitets, init_states_etscalc)
        # assert np.allclose(e_fitets, e_etscalc)
        # assert np.allclose(amse_fitets, amse_etscalc)
        # assert np.isclose(lik_fitets, lik_etscalc)
        print(f"{n}, {time_fitets}, {time_etscalc}")

    return


if __name__ == "__main__":
        test_ets_comparison()
