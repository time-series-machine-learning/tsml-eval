from base import BasePLA
import numpy as np
import sys
import BottomUp

__maintainer__ = []
__all__ = ["SWAB"]

class SWAB(BasePLA):
    
    def __init__(self, max_error, seg_num = 6):
        self.seg_num = seg_num
        self.bottomup = BottomUp(max_error)
        super().__init__(max_error)
        

    def swab(self, time_series):
        seg_ts = []
        buffer = np.empty(self.seg_num, dtype=object)
        sw_lower_bound = len(buffer) / 2
        sw_upper_bound = len(buffer) * 2
        while len(buffer) < 3:
            t = self.bottomup(time_series)
            seg_ts.append(t[0])
            buffer = buffer[len(t) - 1:]
        return None
    
    
    #finds the next potential segment
    def best_line(self, time_series, current_data_point, sw_lower_bound, sw_upper_bound):
        seg_ts = []
        error = 0
        while error < self.max_error:
            seg_ts.append = time_series[current_data_point]
            error = self.calculate_error(seg_ts)
            current_data_point = current_data_point + 1
        return seg_ts
    