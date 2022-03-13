from argparse import _AppendAction
import numpy as np
import time
from mpi_tools import mpi_min,mpi_max,mpi_mean,proc_id,mpi_statistics_scalar,mpi_sum

class Metric:
    def __init__(self,name,capacity):
        self.keys=[]
        self.name=name
        self.windows=[]
        self.capacity=capacity
        self.index=0
        self._sum=0
        self._ctr=0
        self._value=None
            
    def append(self,value):
        self._sum+=value
        self._ctr+=1
        self._value=value
        if len(self.windows)<self.capacity:
            self.windows.append(value)
            self.index+=1
        else:
            self.index%=self.capacity
            self.windows[self.index]=value
            self.index+=1
    
    @property
    def ctr(self):
        return mpi_sum(self._ctr)
    
    @property
    def sum(self):
        return mpi_sum(self._sum)
    
    @property
    def mean(self):
        s,c=mpi_sum([self._sum,self._ctr])    
        assert c!=0
        v=s/c
        return v
    
    @property
    def rmean(self):  
        v=mpi_mean(self.windows)
        return v
    
    @property
    def rmax(self):
        return mpi_max(self.windows)
    
    @property
    def rmin(self):
        return mpi_min(self.windows)
    
    def rstat(self,with_min_and_max=False):
        return mpi_statistics_scalar(self.windows,with_min_and_max)
    
    @property
    def value(self):
        return self._value
    
    add=append
    update=append
    
class Metrics:
    def __init__(self,names,capacities):
        if type(capacities)==int:
            capacities=len(names)*[capacities]
        assert len(names)==len(capacities)
        self.metrics={name:Metric(name,capacity) for name,capacity in zip(names,capacities)}
    
    def __getattr__(self,name):
        return self.metrics[name]
        
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
        print(metric.windows)