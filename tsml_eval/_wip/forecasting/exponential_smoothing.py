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

completely undocumented
"""

def fit_ets(y, n, x, m, error, trend, season, alpha, beta, gamma, phi, e, amse, nmse):
    """Exponential smooting (fit?)

    Do parameters map to Hyndman?? Why 14 not 15?

    Parameters
    ----------

    Returns
    -------
    """
    pass
