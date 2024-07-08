"""Implementation of Hyndman C functions for exponential smoothing in numba.

Three functions from here
https://github.com/robjhyndman/forecast/blob/master/src/etscalc.c

// Functions called by R
void etscalc(double *, int *, double *, int *, int *, int *, int *,
    double *, double *, double *, double *, double *, double *, double *, int*);
void etssimulate(double *, int *, int *, int *, int *,
    double *, double *, double *, double *, int *, double *, double *);
void etsforecast(double *, int *, int *, int *, double *, int *, double *);


Nixtla hove python versions which are straight copies

https://github.com/Nixtla/statsforecast/blob/main/statsforecast/ets.py

completely undocumented. We need to verify what each of the parameters mean,
and check translation.
"""
import math

<<<<<<< HEAD
from numba import njit
import numpy as np


=======
import numpy as np

TOL = 1.0e-10
HUGEN = 1.0e10
>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))
NA = -99999.0
MAX_NMSE = 30
MAX_SEASONAL_PERIOD = 24

<<<<<<< HEAD
@njit
=======

>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))
def fit_ets(y, n, x, m, error, trend, season, alpha, beta, gamma, phi, e, lik, amse, nmse):
    """Exponential smooting (fit?)

    Do parameters map to Hyndman?? Why 14 not 15?

    Parameters
    ----------
    y : np.ndarray
        Time series data.
    n : int
        The length of the time series.
    x : np.ndarray
        Initial states of the ETS model. Starting values for the level, trend, and seasonal
        components. This variable evolves during execution to store the states at each time
        step (i.e., the state space matrix).
    m : int
        The period of the seasonality (e.g., for quaterly data m = 4)
    error : int
        The type of error model (0 -> None, 1 -> additive, 2 -> multiplicative).
    trend : int
        The type of trend model (0 -> None, 1 -> additive, 2 -> multiplicative).
    season : int
        The type of seasonality model (0 -> None, 1 -> additive, 2 -> multiplicative).
    alpha : float
        Smoothing parameter for the level.
    beta : float
        Smoothing parameter for the trend.
    gamma : float
        Smoothing parameter for the seasonality.
    phi : float
        Damping parameter.
    e : np.ndarray
        Residuals of the fitted model.
    lik : np.ndarray
        Likelihood measure.
    amse : np.ndarray
        Empty array for storing the Average Mean Squared Error.
    nmse : int
        The number of steps ahead to be considered for the calculation of AMSE. Determines
        the forcasting horizon.
<<<<<<< HEAD
    y : np.ndarray
        Time series data.
    n : int
        The length of the time series.
    x : np.ndarray
        Initial states of the ETS model. Starting values for the level, trend, and seasonal
        components. This variable evolves during execution to store the states at each time
        step (i.e., the state space matrix).
    m : int
        The period of the seasonality (e.g., for quaterly data m = 4)
    error : int
        The type of error model (0 -> None, 1 -> additive, 2 -> multiplicative).
    trend : int
        The type of trend model (0 -> None, 1 -> additive, 2 -> multiplicative).
    season : int
        The type of seasonality model (0 -> None, 1 -> additive, 2 -> multiplicative).
    alpha : float
        Smoothing parameter for the level.
    beta : float
        Smoothing parameter for the trend.
    gamma : float
        Smoothing parameter for the seasonality.
    phi : float
        Damping parameter.
    e : np.ndarray
        Residuals of the fitted model.
    lik : np.ndarray
        Likelihood measure.
    amse : np.ndarray
        Empty array for storing the Average Mean Squared Error.
    nmse : int
        The number of steps ahead to be considered for the calculation of AMSE. Determines
        the forcasting horizon.
=======
>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))

    Returns
    -------
    """
    assert (m <= MAX_SEASONAL_PERIOD) or (season == 0), "Seasonal period must be <= 24 if seasonality is enabled"
    if m < 1:
        m = 1
    if nmse > MAX_NMSE:
        nmse = MAX_NMSE

    olds = np.zeros(MAX_SEASONAL_PERIOD)
    s = np.zeros(MAX_SEASONAL_PERIOD)
    f = np.zeros(MAX_NMSE)
    denom = np.zeros(MAX_NMSE)
    nstates = 1 + (trend > 0) + m*(season > 0)
    lik[0] = 0.0
    lik2 = 0.0
    denom = np.zeros(nmse)

    l = x[0]
    if trend > 0:
        b = x[1]
    else:
        b = 0.0
    if season > 0:
        for j in range(m):
            s[j] = x[(trend > 0) + j + 1]

    for i in range(n):
        # Copy previous state.
        oldl = l
        if trend > 0:
            oldb = b
        if season > 0:
            for j in range(m):
                olds[j] = s[j]

        # One step forecast.
        forecast(oldl, oldb, olds, m, trend, season, phi, f, nmse)
<<<<<<< HEAD
        if(math.fabs(f[0] - NA) < 1.0e-10):  # TOL
=======
        if(math.fabs(f[0] - NA) < TOL):
>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))
            lik[0] = NA
            return

        if error == 1:  # Additive error model.
            e[i] = y[i] - f[0]
        else:
            e[i] = (y[i] - f[0]) / f[0]

        for j in range(nmse):
            if i+j < n:
                denom[j] += 1.0
                tmp = y[i+j] - f[j]
                amse[j] = (amse[j] * (denom[j]-1)+(tmp*tmp)) / denom[j]

        # Update state.
        l, b, s = update(oldl, l, oldb, b, olds, s, m, trend, season, alpha, beta, gamma, phi, y[i])

        # Store new state.
        x[nstates*(i+1)] = l
        if trend > 0:
            x[nstates*(i+1)+1] = b
        if season > 0:
            for j in range(m):
                x[(trend > 0) + nstates*(i+1) + j + 1] = s[j]
        lik[0] = lik[0] + e[i]*e[i]
        lik2 += np.log(math.fabs(f[0]))

    lik[0] = n * np.log(lik[0])
    if error == 2:  # Multiplicative error model.
        lik[0] = lik[0] + 2*lik2


<<<<<<< HEAD
@njit
=======
>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))
def forecast(l, b, s, m, trend, season, phi, f, h):
    """Performs forcasting.

    Helper function for fit_ets.

    Parameters
    ----------
    l : float
        Current level.
    b : float
        Current trend.
    s : float
        Current seasonal components.
    m : int
        The period of the seasonality (e.g., for quaterly data m = 4)
    trend : int
        The type of trend model (0 -> None, 1 -> additive, 2 -> multiplicative).
    season : int
        The type of seasonality model (0 -> None, 1 -> additive, 2 -> multiplicative).
    phi : float
        Damping parameter.
    f : np.ndarray
        Array to store forcasted values.
    h : int
        The number of steps ahead to forcast.
    """
    phistar = phi

    # Forecasts
    for i in range(h):
        if trend == 0:  # No trend component.
            f[i] = l
        elif trend == 1:  # Additive trend component.
            f[i] = l + phistar*b
        elif b < 0:
            f[i] = NA
        else:
            f[i] = l * b**phistar

        j = m - 1 - i
        while j < 0:
            j += m

        if season == 1:  # Additive seasonal component.
            f[i] = f[i] + s[j]
        elif season == 2:  # Multiplicative seasonal component.
            f[i] = f[i] * s[j]
        if i < (h-1):
<<<<<<< HEAD
            if math.fabs(phi-1) < 1.0e-10:  # TOL
=======
            if math.fabs(phi-1) < TOL:
>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))
                phistar = phistar + 1
            else:
                phistar = phistar + phi**(i+1)


<<<<<<< HEAD
@njit
=======
>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))
def update(oldl, l, oldb, b, olds, s, m, trend, season, alpha, beta, gamma, phi, y):
    """Updates states.

    Helper function for fit_ets

    Parameters
    ----------
    oldl : float
        Previous level.
    l : float
        Current level.
    oldb : float
        Previous trend.
    b : float
        Current trend.
    olds : np.ndarray
        Previous seasonal components.
    s : np.ndarray
        Current seasonal components.
    m : int
        The period of the seasonality (e.g., for quaterly data m = 4)
    trend : int
        The type of trend model (0 -> None, 1 -> additive, 2 -> multiplicative).
    season : int
        The type of seasonality model (0 -> None, 1 -> additive, 2 -> multiplicative).
    alpha : float
        Smoothing parameter for the level.
    beta : float
        Smoothing parameter for the trend.
    gamma : float
        Smoothing parameter for the seasonality.
    phi : float
        Damping parameter.
    y : np.ndarray
        Time series data.

    Returns
    ----------
    l : float
        Updated level.
    b : float
        Updated trend.
    s : float
        Updated seasonal components.
    """
    # New level.
    if trend == 0:  # No trend component.
        phib = 0
        q = oldl   # l(t-1)
    elif trend == 1:  # Additive trend component.
        phib = phi*(oldb)
        q = oldl + phib   # l(t-1) + phi*b(t-1)
<<<<<<< HEAD
    elif math.fabs(phi-1) < 1.0e-10:  # TOL
=======
    elif math.fabs(phi-1) < TOL:
>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))
        phib = oldb
        q = oldl * oldb   # l(t-1) * b(t-1)
    else:
        phib = oldb**phi
        q = oldl * phib   # l(t-1) * b(t-1)^phi

    if season == 0:  # No seasonal component.
        p = y
    elif season == 1:  # Additive seasonal component.
        p = y - olds[m-1]   # y[t] - s[t-m]
    else:
<<<<<<< HEAD
        if math.fabs(olds[m-1]) < 1.0e-10:  # TOL
            p = 1.0e10  # HUGEN
=======
        if math.fabs(olds[m-1]) < TOL:
            p = HUGEN
>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))
        else:
            p = y / olds[m-1]   # y[t] / s[t-m]
    l = q + alpha*(p-q)

    # New growth.
    if trend > 0:
        if trend == 1:  # Additive trend component.
            r = l - oldl   # l[t] - l[t-1]
        else:  # Multiplicative trend component.
<<<<<<< HEAD
            if math.fabs(oldl) < 1.0e-10:  # TOL
                r = 1.0e10  # HUGEN
=======
            if math.fabs(oldl) < TOL:
                r = HUGEN
>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))
            else:
                r = l / oldl   # l[t] / l[t-1]
        b = phib + (beta / alpha)*(r - phib)   # b[t] = phi*b[t-1] + beta*(r - phi*b[t-1])
                                               # b[t] = b[t-1]^phi + beta*(r - b[t-1]^phi)

    # New season.
    if season > 0:
        if season == 1:  # Additive seasonal component.
            t = y - q
        else:  # Multiplicative seasonal compoenent.
<<<<<<< HEAD
            if math.fabs(q) < 1.0e-10:
                t = 1.0e10
=======
            if math.fabs(q) < TOL:
                t = HUGEN
>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))
            else:
                t = y / q
        s[0] = olds[m-1] + gamma*(t - olds[m-1])  # s[t] = s[t-m] + gamma*(t - s[t-m])
        for j in range(m):
            s[j] = olds[j-1]   # s[t] = s[t]

    return l, b, s
<<<<<<< HEAD
=======


if __name__ == "__main__":
    # Example
    y = np.random.randint(10, 15, size=10)
    n = len(y)
    m = 4
    error = 1
    trend = 1
    season = 1
    alpha = 0.2
    beta = 0.1
    gamma = 0.1
    phi = 0.98
    e = np.zeros(n)
    lik = np.zeros(1)
    amse = np.zeros(MAX_NMSE)
    nmse = 3

    # Initial states
    l0 = y[0]
    b0 = (y[1] - y[0]) / 2
    s0 = [y[i] - l0 for i in range(m)]
    x = np.array([l0, b0] + s0, dtype=float)

    # Expand x to hold the states for each time step
    nstates = 1 + (trend > 0) + m * (season > 0)
    x = np.concatenate([x, np.zeros(n * nstates)]).flatten()  # fit_ets expects 'x' to be a one dimensional array

    fit_ets(y, n, x, m, error, trend, season, alpha, beta, gamma, phi, e, lik, amse, nmse)

    print("Residuals (e):", e)
    print("Likelihood (lik):", lik)
    print("AMSE:", amse)
    print("State matrix (x):", x.reshape(-1, nstates))
>>>>>>> 063e03e (ported etscalc, forcast, and update functions. Not yet tested correctness of output. (#248))
