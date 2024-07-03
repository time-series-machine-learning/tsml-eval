
from base import BasePLA
import numpy as np
__maintainer__ = []
__all__ = ["SlidingWindow"]

class SlidingWindow(BasePLA):
    """Piecewise Linear Sliding Window.

    Uses a sliding window algorithm to traverse the dataset in an online manner.

    Parameters
    ----------
    max_error: float
        The maximum error valuefor the function to find before segmenting the dataset

    References
    ----------
    .. [1] Keogh, E., Chu, S., Hart, D. and Pazzani, M., 2001, November. 
    An online algorithm for segmenting time series. (pp. 289-296).
    """
    
    def __init__(self, max_error):
        super().__init__(max_error)
        
    #! clean this up, the while loops are not done in a good manner. This is from the pseudocode
    def transform(self, time_series):
        """Transform a time series

        Parameters
        ----------
        time_series : np.array
            1D time series to be transformed.

        Returns
        -------
        list
            List of transformed segmented time series
        """
        
        seg_ts = []
        anchor = 0
        while anchor < len(time_series): 
            i = 2
            while anchor + i -1 < len(time_series) and self.calculate_error(time_series[anchor:anchor + i]) < self.max_error:
                i = i + 1
            seg_ts.append(self.create_segment(time_series[anchor:anchor + i - 1]))
            anchor = anchor + i - 1
        return seg_ts
    
    
    def transform_flatten(self, time_series):
        """Transform a time series and return a 1d array

        Parameters
        ----------
        time_series : np.array
            1D time series to be transformed.

        Returns
        -------
        list
            List of a flattened transformed time series
        """
        
        pla_timeseries =  self.transform(time_series)
        print(pla_timeseries)
        return np.concatenate(pla_timeseries)
        
    