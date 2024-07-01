
from base import BasePLA
import numpy as np
__maintainer__ = []
__all__ = ["SlidingWindow"]

class SlidingWindow(BasePLA):
    
    def __init__(self, max_error):
        super().__init__(max_error)
        
    """work in progress
    def sliding_window(self, time_series):
        seg_ts = []
        anchor = 0
        for i in range(1, len(time_series)):
            if self.calculate_error(time_series[anchor:i]) > self.max_error:
                seg_ts.append(self.create_segment(time_series[anchor: i - 1]))
                anchor = i - 1
        if(anchor < i):
            seg_ts.append(self.create_segment(time_series[anchor: i - 1]))
        return np.concatenate(seg_ts) """
        
    #! clean this up, the while loops are not done in a good manner. This is from the pseudocode
    def sliding_window(self, time_series):
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
        results = self.sliding_window(time_series)
        dense_array = np.zeros(len(results) - 1)
        for i in range(results - 1):
            dense_array[i] = len(results[i])
        return dense_array            
        
    