from base import BasePLA
import numpy as np
import sys
from _bu import BottomUp

__maintainer__ = []
__all__ = ["SWAB"]

class SWAB(BasePLA):
    
    def __init__(self, max_error, seg_num = 6):
        self.seg_num = seg_num
        self.bottomup = BottomUp(max_error)
        super().__init__(max_error)
        

    def swab(self, time_series):
        seg_ts = []
        seg = self.best_line(time_series, 0)
        current_data_point = len(seg)
        buffer = np.array(seg)
        while len(buffer) > 0:
            t = self.bottomup.bottomUp(time_series)
            seg_ts.append(t[0])
            buffer = buffer[len(t[0]):]
            if(current_data_point != len(time_series)):
                seg = self.best_line(time_series, current_data_point)
                current_data_point = current_data_point + len(seg)
                buffer = np.append(buffer, seg)
        return seg_ts
    
    
    #finds the next potential segment
    def best_line(self, time_series, current_data_point):
        seg_ts = np.array([])
        error = 0
        while current_data_point < len(time_series) and error < self.max_error:
            seg_ts = np.append(seg_ts, time_series[current_data_point])
            error = self.calculate_error(seg_ts)
            current_data_point = current_data_point + 1
        return seg_ts
    
    def dense(self, time_series):
        results = self.swab(time_series)
        dense_array = np.zeros(len(results) - 1)
        segmentation_point = 0
        for i in range(len(results) - 1):
            segmentation_point = segmentation_point + len(results[i])
            dense_array[i] = segmentation_point
        return dense_array    