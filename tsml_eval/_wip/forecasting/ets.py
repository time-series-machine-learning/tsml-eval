from tsml_eval._wip.forecasting.base import BaseForecaster


def ETSForecaster(BaseForecaster):


    def __init__(self, horizon=1, window=None):
        self.horizon = horizon
        self.window = window
        self._is_fitted = False
        super().__init__()

    def fit(self, X):
        """Fit forecaster to series X.

        y : np.ndarray
            Time series data.
        n : int
            The length of the time series.

        x : np.ndarray
            Initial states of the ETS model. Starting values for the level, trend, and seasonal
            components. This variable evolves during execution to store the states at each time
            step (i.e., the state space matrix).

        Returns
        -------
        self
            Fitted estimator
        """
        init_states, m, error, trend, season, alpha, beta, gamma, phi, e, \
            lik_fitets, amse, nmse = _setup(X, len(X))
        l,b,s= _fit_ets(X, len(X), init_states, m, error, trend, season, alpha, beta,
                    gamma, phi, e, lik_fitets, amse, nmse)
        self.error = error
        self.trend = trend
        self.season = season
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        self.e = e
        self.last_level = l
        self.last_trend = b
        self.last_season=s

    def predict(self, X):
        """
        Returns
        -------
        """
        # Update state.
        forecast(self.last_level, self.last_trend, self.last_season, m, trend, season, phi, f, nmse)

    @staticmethod
    def _setup(y, n):
        y = ap
        n = len(ap)
        init_states = np.zeros(n * (2 + 1))
        init_states[0] = y[0]
        init_states[1] = (y[1] - y[0]) / 2
        m = 12
        error = 1
        trend = 1
        season = 0
        alpha = 0.016763333
        beta = 0.001766333
        gamma = 0.
        phi = 0.
        e = np.zeros(n)
        lik_fitets = np.zeros(1)
        amse = np.zeros(MAX_NMSE)
        nmse = 3
        return init_states, m, error, trend, season, alpha, beta, gamma, phi, e, lik_fitets, amse, nmse




@njit(fastmath=True, cache=True)
def _fit_ets(y, n, x, m, error, trend, season, alpha, beta, gamma, phi, e, lik, amse,
             nmse):
    """Exponential smoothing

    Check parameters map to Hyndman?? Why 14 not 15?

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

    Returns
    -------
    """
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
        if(math.fabs(f[0] - NA) < 1.0e-10):  # TOL
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
    return l, b, s




@njit(fastmath=True, cache=True)
def _forecast_one(l, b, s, m, trend, season, phi):
    """Performs forecasting.

    Helper function for fit_ets.

    Parameters
    ----------
    l : float
        Current level.
    b : float
        Current trend.
    s : np.ndarray
        Current seasonal component.
    trend : int
        The type of trend model (0 -> None, 1 -> additive, 2 -> multiplicative).
    season : int
        The type of seasonality model (0 -> None, 1 -> additive, 2 -> multiplicative).
    phi : float
        Damping parameter.
    """
    f = l
    if trend == 1:  # Additive trend component.
        f = f + b*phi
    else:       # Multiplicative trend component.
        f = f * b**phi
    if season == 1:  # Additive seasonal component.
        f = f + s
    else:  # Multiplicative seasonal component.
        f = f * s
    return f


@njit(fastmath=True, cache=True)
def _forecast(l, b, s, m, trend, season, phi, f, h):
    """Performs forecasting.

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
        Array to store forecasted values.
    h : int
        The number of steps ahead to forecast.
    """
    phistar = phi

    # Forecasts
    for i in range(h):
        if trend == 0:  # No trend component.
            f[i] = l
        elif trend == 1:  # Additive trend component.
            f[i] = l + phistar*b
        else:       # Multiplicative trend component.
            f[i] = l * b**phistar

        j = m - 1 - i
        while j < 0:
            j += m

        if season == 1:  # Additive seasonal component.
            f[i] = f[i] + s[j]
        elif season == 2:  # Multiplicative seasonal component.
            f[i] = f[i] * s[j]
        if i < (h-1):
            if math.fabs(phi-1) < 1.0e-10:  # TOL
                phistar = phistar + 1
            else:
                phistar = phistar + phi**(i+1)


@njit(fastmath=True, cache=True)
def _update(oldl, l, oldb, b, olds, s, m, trend, season, alpha, beta, gamma, phi, y):
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
    elif math.fabs(phi-1) < 1.0e-10:  # TOL
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
        if math.fabs(olds[m-1]) < 1.0e-10:  # TOL
            p = 1.0e10  # HUGEN
        else:
            p = y / olds[m-1]   # y[t] / s[t-m]
    l = q + alpha*(p-q)

    # New growth.
    if trend > 0:
        if trend == 1:  # Additive trend component.
            r = l - oldl   # l[t] - l[t-1]
        else:  # Multiplicative trend component.
            if math.fabs(oldl) < 1.0e-10:  # TOL
                r = 1.0e10  # HUGEN
            else:
                r = l / oldl   # l[t] / l[t-1]
        b = phib + (beta / alpha)*(r - phib)   # b[t] = phi*b[t-1] + beta*(r - phi*b[t-1])
                                               # b[t] = b[t-1]^phi + beta*(r - b[t-1]^phi)

    # New season.
    if season > 0:
        if season == 1:  # Additive seasonal component.
            t = y - q
        else:  # Multiplicative seasonal compoenent.
            if math.fabs(q) < 1.0e-10:
                t = 1.0e10
            else:
                t = y / q
        s[0] = olds[m-1] + gamma*(t - olds[m-1])  # s[t] = s[t-m] + gamma*(t - s[t-m])
        for j in range(m):
            s[j] = olds[j-1]   # s[t] = s[t]

    return l, b, s
