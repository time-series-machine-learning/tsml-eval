from base import BasePLA
import numpy as np
import math
__maintainer__ = []
__all__ = ["BottomUp"]

class BottomUp(BasePLA):
    """
        Bottom-Up Segmentation.

        Uses a bottom-up algorithm to traverse the dataset in an online manner.

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
        merge_cost = []
        for i in range(0, len(time_series), 2):
            seg_ts.append(self.create_segment(time_series[i: i + 2]))
        for i in range(len(seg_ts) - 1):
            merge_cost.append(self.calculate_error(seg_ts[i] + seg_ts[i + 1]))

        merge_cost = np.array(merge_cost)

        while len(merge_cost != 0) and min(merge_cost) < self.max_error:
            if(len(merge_cost) == len(seg_ts)):
                print("error")
            pos = np.argmin(merge_cost)
            seg_ts[pos] = np.concatenate((seg_ts[pos], seg_ts[pos + 1]))
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