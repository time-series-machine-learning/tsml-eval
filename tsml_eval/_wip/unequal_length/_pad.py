"""Padding transformer, pad unequal length time series to max length or fixed length."""

__all__ = ["Padder"]
__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer
from sklearn.utils import check_random_state


class Padder(BaseCollectionTransformer):
    """Pad unequal length time series to equal, fixed length.

    Pads the input dataset to either fixed length (at least as long as the longest
    series) or finds the max length series across all series and channels and
    pads to that with zeroes.

    Parameters
    ----------
    pad_length  : int, "min" or "max", default="min"
        Length to pad the series to. If "min", will pad to the shortest
        series seen in ``fit``. If "max", will pad to the longest series seen in
        ``fit``. If an integer, will pad to that length.
    fill_value : int, str or Callable, default=0
        Value to pad with. Can be a float or a statistic string or an numpy array for
        each time series. Supported statistic strings are "mean", "median", "max",
        "min".
    add_noise : float or None, default=None
        Add noise to the padded values of the series.
        Randomly adds a value between 0 and ``add_noise`` to each padded value if
        float.
        Adds no noise if None.
    error_on_long : bool, default=True
        If True, raise an error if a series is longer than pad_length.
        If False, will ignore series longer than pad_length. As the series
        collection could remain unequal length, a list of numpy arrays will be returned
        instead of a 3D numpy array.
    random_state : int, RandomState instance or None, default=None
        Only used if add_noise is True.

        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Examples
    --------
    >>> from aeon.transformations.collection import Padder
    >>> import numpy as np
    >>> X = []
    >>> for i in range(10): X.append(np.random.random((4, 75 + i)))
    >>> padder = Padder(pad_length=200, fill_value =42)
    >>> X2 = padder.fit_transform(X)
    >>> X2.shape
    (10, 4, 200)
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "removes_unequal_length": True,
    }

    def __init__(self, pad_length="max", fill_value=0, add_noise=False, error_on_long=True, random_state=None):
        self.pad_length = pad_length
        self.fill_value = fill_value
        self.add_noise = add_noise
        self.error_on_long = error_on_long
        self.random_state = random_state

        super().__init__()

        self.set_tags(**{"fit_is_empty": isinstance(pad_length, int)})

    def _fit(self, X, y=None):
        """Fit padding transformer to X and y.

        Calculates the max length in X unless padding length passed as an argument.

        Parameters
        ----------
        X : list of [n_cases] 2D np.ndarray shape (n_channels, length_i)
            where length_i can vary between time series or 3D numpy of equal length
            series
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : reference to self
        """
        if self.pad_length == "min":
            self._pad_length = _get_min_length(X)
        elif self.pad_length == "max":
            self._pad_length = _get_max_length(X)
        elif isinstance(self.pad_length, int):
            pass  # todo fix base class fit_transform and remove this
        else:
            raise ValueError("pad_length must be 'min', 'max' or an integer.")

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : list of [n_cases] 2D np.ndarray shape (n_channels, length_i)
            where length_i can vary between time series or 3D numpy of equal length
            series
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : numpy3D array (n_cases, n_channels, self.pad_length_)
            padded time series from X.
        """
        # Must call fit if pad_length is None
        pad_length = self.pad_length if isinstance(self.pad_length, int) else self._pad_length

        if self.error_on_long:
            max_length = _get_max_length(X)
            if max_length > pad_length:
                raise ValueError(
                    "max length of series in X is greater than the provided pad_length"
                    "(or greater than the series seen in fit if pad_length is None)."
                )

        # Determine if fill value is a function
        func = None
        if isinstance(self.fill_value, str):
            if self.fill_value == "mean":
                func = np.mean
            elif self.fill_value == "median":
                func = np.median
            elif self.fill_value == "min":
                func = np.min
            elif self.fill_value == "max":
                func = np.max
            elif self.fill_value == "last":
                func = lambda x: x[-1]
            else:
                raise ValueError(
                    "Supported str values for fill_value are {mean, median, min, "
                    "max, last}."
                )
        elif callable(self.fill_value):
            func = self.fill_value

        rng = check_random_state(self.random_state)

        # Pad the series
        Xt = []
        for i, series in enumerate(X):
            if series.shape[1] >= pad_length:
                Xt.append(series)
                continue

            # Amount to pad
            pad_width = (0, pad_length - series.shape[1])
            padded_series = []
            for j, channel in enumerate(series):
                # Pad the series channel array
                p = np.pad(
                    channel,
                    pad_width,
                    mode="constant",
                    constant_values=self.fill_value if func is None else func(channel),
                )

                if self.add_noise is not None:
                    p[series.shape[1] :] += rng.uniform(0, self.add_noise, size=pad_length - series.shape[1])

                padded_series.append(p)
            Xt.append(np.array(padded_series))

        return np.array(Xt) if self.error_on_long else Xt


def _get_min_length(X):
    min_length = X[0].shape[1]
    for x in X:
        if x.shape[1] < min_length:
            min_length = x.shape[1]
    return min_length

def _get_max_length(X):
    max_length = X[0].shape[1]
    for x in X:
        if x.shape[1] > max_length:
            max_length = x.shape[1]
    return max_length
