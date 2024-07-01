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
    def segment(self, time_series):
        """Segment a time series

        Parameters
        ----------
        time_series : np.array
            1D time series to be segmented.

        Returns
        -------
        list
            List of segmentations
        """
        
        seg_ts = []
        best_so_far = sys.float_info.max
        breakpoint = None
        for i in range(2, len(time_series -2)):
            improvement_in_approximation = self.improvement_splitting_here(time_series, i)
            if(improvement_in_approximation < best_so_far):
                breakpoint = i
                best_so_far = improvement_in_approximation

        left_segment = time_series[:breakpoint]
        right_segment = time_series[breakpoint:]

        if self.calculate_error(left_segment) > self.max_error:
            seg_ts.append(self.segment(left_segment))
        else:
            seg_ts.extend([left_segment])

        if self.calculate_error(right_segment) > self.max_error:
            seg_ts.append(self.segment(right_segment))
        else:
            seg_ts.extend([right_segment])

        return seg_ts
    
    
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
    
    def dense(self, time_series):
        """Return the dense values of a segmented time series

        Parameters
        ----------
        time_series : np.array
            1D time series to be segmented.

        Returns
        -------
        list
            dense values of a segmentation
        """
        
        results = self.segment(time_series)
        dense_array = np.zeros(len(results) - 1)
        segmentation_point = 0
        for i in range(len(results) - 1):
            segmentation_point = segmentation_point + len(results[i])
            dense_array[i] = segmentation_point
        return dense_array         
    