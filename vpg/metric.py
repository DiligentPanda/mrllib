from argparse import _AppendAction
import numpy as np
import time
from mpi_tools import mpi_min,mpi_max,mpi_mean,proc_id,mpi_statistics_scalar

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
        v=mpi_mean(self.values)
        return v
    
    def max(self):
        return mpi_max(self.values)
    
    def min(self):
        return mpi_min(self.values)
    
    def stat(self,with_min_and_max=False):
        return mpi_statistics_scalar(self.values,with_min_and_max)
    
    def value(self):
        index=(self.index-1+self.capacity)%self.capacity
        return self.values[index]
    
    add=append
    
class Metrics:
    def __init__(self,names,capacities):
        if type(capacities)==int:
            capacities=len(names)*[capacities]
        assert len(names)==len(capacities)
        self.metrics={name:Metric(name,capacity) for name,capacity in zip(names,capacities)}
        
    def append(self,name,value):
        self.metrics[name].append(value)
    
    def min(self,name):
        return self.metrics[name].min()
    
    def mean(self,name):
        return self.metrics[name].mean()
    
    def max(self,name):
        return self.metrics[name].max()
    
    def stat(self,name,with_min_and_max=False):
        return self.metrics[name].stat(with_min_and_max)
    
    def value(self,name):
        return self.metrics[name].value()
    
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