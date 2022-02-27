from argparse import _AppendAction
import numpy as np
import time
from mpi_tools import mpi_min,mpi_max,mpi_avg,proc_id

class Metric:
    def __init__(self,name,capacity):
        self.name=name
        self.values=[]
        self.capacity=capacity
        self.index=0
            
    def append(self,value):
        if len(self.values)<self.capacity:
            self.values.append(value)
            self.index+=1
        else:
            self.index%=self.capacity
            self.values[self.index]=value
            self.index+=1
            
    def mean(self):
        v=np.mean(self.values)
        return v
    
    def max(self):
        return mpi_max(self.values)
    
    def min(self):
        return mpi_min(self.values)
    
    add=append
    
class Metrics:
    def __init__(self,names,capacity):
        self.metrics={name:Metric(name,capacity) for name in names}
        
    def append(self,name,value):
        self.metrics[name].append(value)
    
    def min(self,name):
        return self.metrics[name].min()
    
    def mean(self,name):
        return self.metrics[name].mean()
    
    def max(self,name):
        return self.metrics[name].max()
    
    add=append
        
class Timer:
    def __init__(self):
        self.timestamps={}
    
    def record(self,key):
        self.timestamps[key]=time.time()
    
    def time(self,okey,nkey=None):
        t=time.time()
        if nkey is not None:
            self.timestamps[nkey]=t
        return round(t-self.timestamps[okey],2)

if __name__=="__main__":
    import random
    metric=Metric("test",5)
    for i in range(10):
        metric.append(int(random.random()*10))
        print(metric.values)