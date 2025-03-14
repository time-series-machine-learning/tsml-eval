"""Truncation transformer - truncate unequal length collections."""

__all__ = ["Truncator"]
__maintainer__ = []

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


class Truncator(BaseCollectionTransformer):
    """Truncate unequal length time series to a lower bounds.

    Truncates all series in collection to a selected length. This
    transformer assumes that all series have the same number of channels (dimensions)
    and that all channels in a single series are the same length.

    Parameters
    ----------
    truncated_length : int, "min" or "max", default="min"
        The length to truncate series to. If "min", will truncate to the shortest
        series seen in ``fit``. If "max", will truncate to the longest series seen in
        ``fit``. If an integer, will truncate to that length.
    error_on_short : bool, default=True
        If True, raise an error if a series is shorter than truncated_length.
        If False, will ignore series shorter than truncated_length. As the series
        collection could remain unequal length, a list of numpy arrays will be returned
        instead of a 3D numpy array.

    Examples
    --------
    >>> from aeon.transformations.collection import Truncator
    >>> import numpy as np
    >>> X = []
    >>> for i in range(10): X.append(np.random.random((4, 75 + i)))
    >>> truncator = Truncator(truncated_length=10)
    >>> X2 = truncator.fit_transform(X)
    >>> X2.shape
    (10, 4, 10)

    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "removes_unequal_length": True,
    }

    def __init__(self,
        truncated_length="min",
        error_on_short=True
    ):
        self.truncated_length = truncated_length
        self.error_on_short = error_on_short

        super().__init__()

        self.set_tags(**{"fit_is_empty": isinstance(truncated_length, int)})


    def _fit(self, X, y=None):
        """Fit transformer to X and y.

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
        if self.truncated_length == "min":
            self._truncated_length = _get_min_length(X)
        elif self.truncated_length == "max":
            self._truncated_length = _get_max_length(X)
        elif isinstance(self.truncated_length, int):
            pass  # todo fix base class fit_transform and remove this
        else:
            raise ValueError("truncated_length must be 'min', 'max' or an integer.")


    def _transform(self, X, y=None):
        """Truncate X and return a transformed version.

        Parameters
        ----------
        X : list of [n_cases] 2D np.ndarray shape (n_channels, length_i)
            where length_i can vary between time series.
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : numpy3D array (n_cases, n_channels, self.truncated_length_)
            truncated time series from X.
        """
        # Must call fit unless truncated_length is an int
        truncated_length = self.truncated_length if isinstance(self.truncated_length, int) else self._truncated_length

        if self.error_on_short:
            min_length = _get_min_length(X)
            if min_length < truncated_length:
                raise ValueError(
                    "min length of series in X is less than the provided "
                    "truncated_length (or less than the series seen in fit if "
                    "truncated_length is str)."
                )

        Xt = [x[:, : truncated_length] for x in X]
        return np.array(Xt) if self.error_on_short else Xt

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        params = {"truncated_length": 5}
        return params

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
