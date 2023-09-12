import numpy as np
from numba import njit, int64, prange
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_random_state


def ps(x, sign=1, n=None, pad="mean"):
    """Power spectrum transformer.

    Power spectrum transform, currently calculated using np function.
    It would be worth looking at ff implementation, see difference in speed
    to java.

    Parameters
    ----------
    x : array-like shape = [interval_width]
    sign : {-1, 1}, default = 1
    n : int, default=None
    pad : str or function, default='mean'
        controls the mode of the pad function
        see numpy.pad for more details
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Returns
    -------
    y : array-like shape = [len(x)/2]
    """
    x_len = x.shape[-1]
    x_is_1d = x.ndim == 1
    # pad or slice series if length is not of power of 2 or n is specified
    if x_len & (x_len - 1) != 0 or n:
        # round n (or the length of x) to next power of 2
        # when n is not specified
        if not n:
            n = _round_to_nearest_power_of_two(x_len)
        # pad series up to n when n is larger otherwise slice series up to n
        if n > x_len:
            pad_length = (0, n - x_len) if x_is_1d else ((0, 0), (0, n - x_len))
            x_in_power_2 = np.pad(x, pad_length, mode=pad)
        else:
            x_in_power_2 = x[:n] if x_is_1d else x[:, :n]
    else:
        x_in_power_2 = x
    # use sign to determine inverse or normal fft
    # using the norm in numpy fft function
    # backward = normal fft, forward = inverse fft (divide by n after fft)
    # note: use the following code when upgrade numpy to 1.20
    # norm = "backward" if sign > 0 else "forward"
    # fft = np.fft.rfft(x_in_power_2, norm=norm)
    if sign < 0:
        x_in_power_2 /= n
    fft = np.fft.rfft(x_in_power_2)
    fft = fft[:-1] if x_is_1d else fft[:, :-1]
    return np.abs(fft)


@njit("int64(int64)", cache=True)
def _round_to_nearest_power_of_two(n):
    return int64(1 << round(np.log2(n)))


@njit(parallel=True, cache=True)
def acf(x, max_lag):
    """Autocorrelation function transform.

    currently calculated using standard stats method. We could use inverse of power
    spectrum, especially given we already have found it, worth testing for speed and
    correctness. HOWEVER, for long series, it may not give much benefit, as we do not
    use that many ACF terms.

    Parameters
    ----------
    x : array-like shape = [interval_width]
    max_lag: int
        The number of ACF terms to find.

    Returns
    -------
    y : array-like shape = [max_lag]
    """
    y = np.empty(max_lag)
    length = len(x)
    for lag in prange(1, max_lag + 1):
        # Do it ourselves to avoid zero variance warnings
        lag_length = length - lag
        x1, x2 = x[:-lag], x[lag:]
        s1 = np.sum(x1)
        s2 = np.sum(x2)
        m1 = s1 / lag_length
        m2 = s2 / lag_length
        ss1 = np.sum(x1 * x1)
        ss2 = np.sum(x2 * x2)
        v1 = ss1 - s1 * m1
        v2 = ss2 - s2 * m2
        v1_is_zero, v2_is_zero = v1 <= 1e-9, v2 <= 1e-9
        if v1_is_zero and v2_is_zero:  # Both zero variance,
            # so must be 100% correlated
            y[lag - 1] = 1
        elif v1_is_zero or v2_is_zero:  # One zero variance
            # the other not
            y[lag - 1] = 0
        else:
            y[lag - 1] = np.sum((x1 - m1) * (x2 - m2)) / np.sqrt(v1 * v2)
        # _x = np.vstack((x[:-lag], x[lag:]))
        # s = np.sum(_x, axis=1)
        # ss = np.sum(_x * _x, axis=1)
        # v = ss - s * s / l
        # zero_variances = v <= 1e-9
        # i = lag - 1
        # if np.all(zero_variances):  # Both zero variance,
        #     # so must be 100% correlated
        #     y[i] = 1
        # elif np.any(zero_variances):  # One zero variance
        #     # the other not
        #     y[i] = 0
        # else:
        #     m = _x - s.reshape(2, 1) / l
        #     y[i] = (m[0] @ m[1]) / np.sqrt(np.prod(v))

    return y

def _acf(X, istart, iend, lag):
    n_instances, _ = X.shape
    acf_x = np.empty(shape=(n_instances, lag))
    for j in range(n_instances):
        interval_x = X[j, istart:iend]
        acf_x[j] = acf(interval_x, lag)

    return acf_x


def _ps(X, istart, iend, lag):
    n_instances, _ = X.shape
    ps_len = _round_to_nearest_power_of_two(istart - iend)
    ps_x = np.empty(shape=(n_instances, ps_len))
    for j in range(n_instances):
        interval_x = X[j, istart:iend]
        ps_x[j] = ps(interval_x, n=ps_len * 2)

    return ps_x


FEATURE_CANDIDATES = [_acf, _ps]


class IntervalSplitter:
    """RISE-based splitter for TS-CHIEF implementation."""

    @staticmethod
    def generate(X, y, random_state=None):
        """Generate a randomized interval splitter candidate."""
        samples, dims, length = X.shape
        splitter = IntervalSplitter()
        splitter.rng = check_random_state(random_state)
        splitter.dim = splitter.rng.randint(dims)

        min_interval = min(16, length)
        acf_min_values = 4

        splitter.istart = splitter.rng.randint(0, length - min_interval)
        splitter.iend = splitter.rng.randint(splitter.istart + min_interval, length)
        acf_lag = 100
        if acf_lag > splitter.iend - splitter.istart - acf_min_values:
            acf_lag = splitter.iend - splitter.istart - acf_min_values
        if acf_lag < 0:
            acf_lag = 1

        splitter.acf_lag = acf_lag

        splitter.transform = splitter.rng.choice(FEATURE_CANDIDATES)
        X_transformed = splitter.transform(
            X[:, splitter.dim, :], splitter.istart, splitter.iend, splitter.acf_lag
        )

        splitter.tree = DecisionTreeClassifier(
            criterion="gini", max_depth=1, max_features=None, random_state=splitter.rng
        ).fit(X_transformed, y)

        return splitter

    def split(self, X):
        """Split incoming data."""
        X = X[:, self.dim, :]
        X_transformed = self.transform(X, self.istart, self.iend, self.acf_lag)

        return self.tree.apply(X_transformed) - 1
