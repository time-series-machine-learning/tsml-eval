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
void etscalc(double *y, int *n, double *x, int *m, int *error, int *trend, int *season,
    double *alpha, double *beta, double *gamma, double *phi, double *e, double *lik, double *amse, int *nmse)
y is a double array
n the length of y
I think the rest of the parameters are scalars
    int i, j, nstates;
    double oldl, l, oldb, b, olds[24], s[24], f[30], lik2, tmp, denom[30];
NIXTLA:
    oldb = 0.0
    olds = np.zeros(max(24, m))
    s = np.zeros(max(24, m))
    f = np.zeros(30)
    denom = np.zeros(30)


    if((*m > 24) & (*season > NONE))
        return;
    else if(*m < 1)
        *m = 1;
    if(*nmse > 30)
        *nmse = 30;
    nstates = (*m)*(*season>NONE) + 1 + (*trend>NONE);
NIXTLA
    if m < 1:
        m = 1
    if nmse > 30:
        nmse = 30
    nstates = m * (season > NONE) + 1 + (trend > NONE)
Lets use 0 instead of NONE

    // Copy initial state components
    l = x[0];
    if(*trend > NONE)
        b = x[1];
    if(*season > NONE){
        for(j=0; j<(*m); j++)
            s[j] = x[(*trend>NONE)+j+1];
    }
    *lik = 0.0;
    lik2 = 0.0;
    for(j=0; j<(*nmse); j++)
    {
        amse[j] = 0.0;
        denom[j] = 0.0;
    }
NIXTLA
    # Copy initial state components
    l = x[0]
    if trend > NONE:
        b = x[1]
    else:
        b = 0.0
    if season > NONE:
        for j in range(m):
            s[j] = x[(trend > NONE) + j + 1]
    lik = 0.0
    lik2 = 0.0
    for j in range(nmse):
        amse[j] = 0.0
        denom[j] = 0.0


Main loop here?
    for (i=0; i<(*n); i++)
    {
        // COPY PREVIOUS STATE
        oldl = l;
        if(*trend > NONE)
            oldb = b;
        if(*season > NONE)
        {
            for(j=0; j<(*m); j++)
                olds[j] = s[j];
        }

        // ONE STEP FORECAST
        forecast(oldl, oldb, olds, *m, *trend, *season, *phi, f, *nmse);


        if(fabs(f[0]-NA) < TOL)
        {
            *lik = NA;
            return;
        }

        if(*error == ADD)
            e[i] = y[i] - f[0];
        else
            e[i] = (y[i] - f[0])/f[0];
        for(j=0; j<(*nmse); j++)
        {
            if(i+j<(*n))
            {
                denom[j] += 1.0;
                tmp = y[i+j]-f[j];
                amse[j] = (amse[j] * (denom[j]-1.0) + (tmp*tmp)) / denom[j];
            }
        }

        // UPDATE STATE
        update(&oldl, &l, &oldb, &b, olds, s, *m, *trend, *season, *alpha, *beta, *gamma, *phi, y[i]);

        // STORE NEW STATE
        x[nstates*(i+1)] = l;
        if(*trend > NONE)
            x[nstates*(i+1)+1] = b;
        if(*season > NONE)
        {
           for(j=0; j<(*m); j++)
                x[(*trend>NONE)+nstates*(i+1)+j+1] = s[j];
        }
        *lik = *lik + e[i]*e[i];
        lik2 += log(fabs(f[0]));
    }
    *lik = (*n) * log(*lik);
    if(*error == MULT)
        *lik += 2*lik2;
}
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
