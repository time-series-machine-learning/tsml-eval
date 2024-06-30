from base import BasePLA
import numpy as np
__maintainer__ = []
__all__ = ["BottomUp"]

class BottomUp(BasePLA):
    
    def __init__(self, max_error):
        super().__init__(max_error)
    
    #clean the code
    def bottomUp(self, time_series):
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