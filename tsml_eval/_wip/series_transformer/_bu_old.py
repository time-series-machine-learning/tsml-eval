from base import BasePLA
import numpy as np
import math
__maintainer__ = []
__all__ = ["BottomUp"]

class BottomUp(BasePLA):
    """
        Piecewise Linear Bottom-Up.

        Uses a bottom-up algorithm to traverse the dataset in an offline manner.

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
    
    #clean the code
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
        merge_cost = []
        for i in range(0, len(time_series), 2):
            seg_ts.append(self.create_segment(time_series[i: i + 2]))
        for i in range(len(seg_ts) - 1):
            merge_cost.append(self.calculate_error(seg_ts[i] + seg_ts[i + 1]))

        merge_cost = np.array(merge_cost)

        while len(merge_cost) != 0 and min(merge_cost) < self.max_error:
            pos = np.argmin(merge_cost)
            seg_ts[pos] = self.create_segment(np.concatenate((seg_ts[pos], seg_ts[pos + 1])))
            seg_ts.pop(pos + 1)
            if (pos + 1) < len(merge_cost):
                merge_cost = np.delete(merge_cost, pos + 1)
            else:
                merge_cost= np.delete(merge_cost, pos)

            if pos != 0:
                merge_cost[pos - 1] = self.calculate_error(np.concatenate((seg_ts[pos - 1], seg_ts[pos])))

            if((pos + 1) < len(seg_ts)):
                merge_cost[pos] = self.calculate_error(np.concatenate((seg_ts[pos], seg_ts[pos + 1])))

    
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
        return np.concatenate(pla_timeseries)