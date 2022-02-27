import numpy as np
from tools import discount_cumsum,to_tensor,to_array
import torch.nn as nn
import torch

class VPGBuffer:
    def __init__(self,params):
        self.capacity=params["buffer_capacity"]
        self.t_buffer=[]
        self.state_shape=list(params["state_shape"])
        self.gamma=params["gamma"]
        self.states=np.zeros([self.capacity,]+self.state_shape,dtype=np.float)
        self.actions=np.zeros(self.capacity,dtype=np.int)
        self.rt_vals=np.zeros(self.capacity,dtype=np.float)
        self.ad_vals=np.zeros(self.capacity,dtype=np.float)
        self.logprobs=np.zeros(self.capacity,dtype=np.float)
        self.index=0
    
    def append(self,state,action,reward,terminal,logprob):
        if not terminal:
            self.t_buffer.append((state,action,reward,logprob))
        else:
            # calculate return values
            states,actions,rewards,logprobs=zip(*self.t_buffer)
            states=np.array(states,np.float32)
            actions=np.array(actions,np.int)
            rewards=np.array(rewards,np.float32)
            logprobs=np.array(logprobs,np.float32)
            rt_vals=discount_cumsum(rewards,self.gamma)
            
            # append data in this tracjectory
            n=len(rt_vals)
            sidx=self.index
            eidx=self.index+n
            if eidx>self.capacity:
                eidx=self.capacity
                n=self.capacity-sidx
                indices=np.arange(len(rt_vals))
                np.random.shuffle(indices)
                indices=indices[:n]
                state=state[indices]
                action=action[indices]
                rt_vals=rt_vals[indices]
                logprobs=logprobs[indices]
            self.states[sidx:eidx]=states
            self.actions[sidx:eidx]=actions
            self.rt_vals[sidx:eidx]=rt_vals
            self.logprobs[sidx:eidx]=logprobs
            self.index=eidx
            
            # clean temporary buffer
            self.t_buffer=[]

    def size(self):
        return self.index

    def reset(self):
        self.index=0
        
    def get_data(self):
        return self.states[:self.index],self.actions[:self.index],self.rt_vals[:self.index],self.logprobs[:self.index]