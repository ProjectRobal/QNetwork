import numpy as np


class TrendBuffer:
    def __init__(self,size:int) -> None:
        self.buffer:np.ndarray=np.zeros(size,dtype=np.float32)
        self.timeseries:np.ndarray=np.arange(size,dtype=np.int32)


    def push(self,x:float):

        np.roll(self.buffer,-1)

        self.buffer[-1]=x
    
    def trendline(self)->float:
        output=np.polyfit(self.timeseries,self.buffer,1)
        
        return output[0]
    
    def stdev(self)->float:
        
        return np.std(self.buffer)
    
    def mean(self)->float:

        return np.mean(self.buffer)

