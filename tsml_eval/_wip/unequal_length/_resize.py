"""Time series resizer."""

__all__ = ["Resizer"]
__maintainer__ = []

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer

from tsml_eval._wip.unequal_length._pad import _get_min_length, _get_max_length


class Resizer(BaseCollectionTransformer):
    """Time series interpolator/re-sampler.

    Transformer that resizes series using np.linspace  is fitted on each channel
    independently. After transformation the collection will be a numpy array shape (
    n_cases, n_channels, length). It is not capable of sensibly handling missing
    values.

    Parameters
    ----------
    length : integer, the length of time series to resize to.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection import Resizer
    >>> # Unequal length collection of time series
    >>> X_list = []
    >>> for i in range(10): X_list.append(np.random.rand(5,10+i))
    >>> # Equal length collection of time series
    >>> X_array = np.random.rand(10,3,30)
    >>> trans = Resizer(length = 50)
    >>> X_new = trans.fit_transform(X_list)
    >>> X_new.shape
    (10, 5, 50)
    >>> X_new = trans.fit_transform(X_array)
    >>> X_new.shape
    (10, 3, 50)
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "removes_unequal_length": True,
    }

    def __init__(self, length="max"):
        self.length = length
        super().__init__()

        self.set_tags(**{"fit_is_empty": isinstance(length, int)})


    def _fit(self, X, y=None):
        if self.length == "min":
            self._length = _get_min_length(X)
        elif self.length == "max":
            self._length = _get_max_length(X)
        elif isinstance(self.length, int):
            pass  # todo fix base class fit_transform and remove this
        else:
            raise ValueError("")


    def _transform(self, X, y=None):
        """Fit a linear function on each channel of each series, then resample.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or
            list size [n_cases] of 2D nump arrays, case i has shape (n_channels,
            length_i). Collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        3D numpy array of shape (n_cases, n_channels, self.length)
        """
        length = self.length if isinstance(self.length, int) else self._length

        Xt = []
        for x in X:
            x_new = np.zeros((x.shape[0], length))
            x2 = np.linspace(0, 1, x.shape[1])
            x3 = np.linspace(0, 1, length)
            for i, row in enumerate(x):
                x_new[i] = np.interp(x3, x2, row)
            Xt.append(x_new)
        return np.array(Xt)
