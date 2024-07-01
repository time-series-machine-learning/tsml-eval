
from base import BasePLA
import numpy as np
__maintainer__ = []
__all__ = ["SlidingWindow"]

class SlidingWindow(BasePLA):
    """Sliding Window Segmentation.

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
        anchor = 0
        while anchor < len(time_series): 
            i = 2
            while anchor + i -1 < len(time_series) and self.calculate_error(time_series[anchor:anchor + i]) < self.max_error:
                i = i + 1
            seg_ts.append(self.create_segment(time_series[anchor:anchor + i - 1]))
            anchor = anchor + i - 1
        return seg_ts
    
    
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
        
    