from base import BasePLA
import numpy as np
import sys

__maintainer__ = []
__all__ = ["TopDown"]

class TopDown(BasePLA):
    """
        Top-Down Segmentation.

        Uses a top-down algorithm to traverse the dataset in an online manner.

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
        
    #Implement a cache system for this
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
        
        best_so_far = sys.float_info.max
        breakpoint = None
        
        for i in range(2, len(time_series -2)):
            improvement_in_approximation = self.improvement_splitting_here(time_series, i)
            if(improvement_in_approximation < best_so_far):
                breakpoint = i
                best_so_far = improvement_in_approximation

        left_found_segment = time_series[:breakpoint]
        right_found_segment = time_series[breakpoint:]

        left_segment = None
        right_segment = None

        if self.calculate_error(left_found_segment) > self.max_error:
            left_segment = self.transform(left_found_segment)
        else:
            left_segment = [self.create_segment(left_found_segment)]

        if self.calculate_error(right_found_segment) > self.max_error:
            right_segment = self.transform(right_found_segment)
        else:
            right_segment = [self.create_segment(right_found_segment)]

        return left_segment + right_segment
    
    
    def improvement_splitting_here(self, time_series, breakpoint):
        """Returns the squared sum error of the left and right segment
        splitted off at a particual point in a time series

        Parameters
        ----------
        time_series : np.array
            1D time series.
        breakpoint : int
            the break point within the time series array

        Returns
        -------
        error
            the squared sum error of the split segmentations
        """
        
        left_segment = time_series[:breakpoint]
        right_segment = time_series[breakpoint:]
        return self.calculate_error(left_segment) + self.calculate_error(right_segment)
    
    
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
    