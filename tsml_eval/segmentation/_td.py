from base import BasePLA
import numpy as np
import sys

__maintainer__ = []
__all__ = ["TopDown"]

class TopDown(BasePLA):
    
    def __init__(self, max_error):
        super().__init__(max_error)
        
    #Implement a cache system for this
    def topDown(self, time_series):
       seg_ts = []
        best_so_far = sys.float_info.max
        breakpoint = None
        for i in range(2, len(time_series -2)):
            improvement_in_approximation = self.improvement_splitting_here(time_series, i)
            if(improvement_in_approximation < best_so_far):
                breakpoint = i
                best_so_far = improvement_in_approximation

        if breakpoint == None:
            return [time_series]

        left_segment = time_series[:breakpoint]
        right_segment = time_series[breakpoint:]

        if self.calculate_error(left_segment) > self.max_error:
            seg_ts.extend(self.topDown(left_segment))
        else:
            seg_ts.append(left_segment)


        if self.calculate_error(right_segment) > self.max_error:
            seg_ts.extend(self.topDown(right_segment))
        else:
            seg_ts.append(right_segment)

        return seg_ts
    
    
    def improvement_splitting_here(self, time_series, breakpoint):
        left_segment = time_series[:breakpoint]
        right_segment = time_series[breakpoint:]
        return self.calculate_error(left_segment) + self.calculate_error(right_segment)
        