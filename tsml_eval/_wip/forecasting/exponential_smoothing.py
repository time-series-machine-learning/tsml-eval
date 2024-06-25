"""Working area for implementing Hyndman C functions for exponential smoothing in
Numba.

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
"""

