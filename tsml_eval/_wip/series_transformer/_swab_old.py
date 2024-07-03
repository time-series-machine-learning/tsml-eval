from base import BasePLA
import numpy as np
import sys
from _bu import BottomUp

__maintainer__ = []
__all__ = ["SWAB"]

class SWAB(BasePLA):
    """
        SWAB (Sliding Window And Bottom-Up) Segmentation.

        Uses SWAB algorithm as described in [1] to traverse the dataset in an online manner.

        Parameters
        ----------
        max_error: float
            The maximum error valuefor the function to find before segmenting the dataset

        References
        ----------
        .. [1] Keogh, E., Chu, S., Hart, D. and Pazzani, M., 2001, November. 
        An online algorithm for segmenting time series. (pp. 289-296).
    """
    
    def __init__(self, max_error, sequence_num):
        self.bottomup = BottomUp(max_error)
        self.sequence_num = sequence_num
        super().__init__(max_error)
        
    #need to check buffer, i think it does grow exponantionally large
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
        
        lower_boundary_window = int(self.sequence_num  / 2)
        upper_boundary_window = self.sequence_num * 2
        
        seg = self.best_line(time_series, 0, lower_boundary_window, upper_boundary_window)
        current_data_point = len(seg)
        buffer = np.array(seg)
        
        while len(buffer) > 0:
            t = self.bottomup.transform(time_series)
            seg_ts.append(t[0])
            buffer = buffer[len(t[0]):]
            if(current_data_point >= len(time_series)):
                seg = self.best_line(time_series, current_data_point, lower_boundary_window, upper_boundary_window)
                current_data_point = current_data_point + len(seg)
                buffer = np.append(buffer, seg)
            else:
                buffer = np.array([])
                t = t[1:]
                for i in range(len(t)):
                    seg_ts.append(t[i])
        return seg_ts
    
    
    #finds the next potential segment
    def best_line(self, time_series, current_data_point, lower_boundary_window, upper_boundary_window):
        """Uses sliding window to find the next best segmentation candidate

        Parameters
        ----------
        time_series : np.array
            1D time series to be segmented.
        current_data_point : int
            the current_data_point we are observing
        lower_boundary_window: int
            the lower boundary of the window
        upper_boundary_window: int
            the uppoer boundary of the window
        
        Returns
        -------
        np.array
            new found segmentation candidates
        """
        
        max_window_length = current_data_point + upper_boundary_window
        seg_ts = np.array(time_series[current_data_point: current_data_point + lower_boundary_window])
        current_data_point = current_data_point + lower_boundary_window
        error = 0
        while current_data_point < max_window_length and current_data_point < len(time_series) and error < self.max_error:
            seg_ts = np.append(seg_ts, time_series[current_data_point])
            error = self.calculate_error(seg_ts)
            current_data_point = current_data_point + 1
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