import numpy as np
from tools import discount_cumsum,to_tensor,to_array
import torch.nn as nn
import torch
from gym.spaces import Box,Discrete

class VPGBuffer:
    def __init__(self,params):
        self.params=params
        self.capacity=params["buffer_capacity"]
        self.t_buffer=[]
        self.obs_shape=list(params["observation_space"].shape)
        self.action_shape=list(params["action_space"].shape)
        self.gamma=params["gamma"]
        self.lam=params["lam"]
        self.use_adv=params["use_adv"]
        self.use_gae=params["use_gae"]
        self.states=np.zeros([self.capacity,]+self.obs_shape,dtype=np.float)
        if isinstance(self.params["action_space"],Discrete):
            self.action_dtype=np.int32
        elif isinstance(self.params["action_space"],Box):
            self.action_dtype=np.float32
        self.actions=np.zeros([self.capacity,]+self.action_shape,dtype=self.action_dtype)
        self.rt_vals=np.zeros(self.capacity,dtype=np.float)
        self.adv_vals=np.zeros(self.capacity,dtype=np.float)
        self.logprobs=np.zeros(self.capacity,dtype=np.float)
        self.n=0
    
    def append(self,state,action,reward,value,logprob):
            self.t_buffer.append((state,action,reward,value,logprob))
            self.n+=1
    
    def finish(self,last_val=0):
            # calculate return values
            states,actions,rewards,values,logprobs=zip(*self.t_buffer)
            states=np.array(states,np.float32)
            # todo
            actions=np.array(actions,self.action_dtype)
            rewards=np.array(rewards,np.float32)
            values=np.array(values,np.float32)
            logprobs=np.array(logprobs,np.float32)
            
            rewards=np.append(rewards,last_val)
            values=np.append(values,last_val)
            
            rt_vals=discount_cumsum(rewards,self.gamma)[:-1]
            
            if self.use_adv:
                if not self.use_gae:
                    # MC based advantages
                    adv_vals=rt_vals-values[:-1]
                else:
                    # GAE
                    deltas=rewards[:-1]+self.gamma*values[1:]-values[:-1]
                    adv_vals=discount_cumsum(deltas,self.gamma*self.lam)
            else:
                adv_vals=rt_vals
            
            # append data in this tracjectory
            n=len(rt_vals)
            sidx=self.index
            eidx=self.index+n
            assert eidx<self.capacity
            self.states[sidx:eidx]=states
            self.actions[sidx:eidx]=actions
            self.adv_vals[sidx:eidx]=adv_vals
            self.rt_vals[sidx:eidx]=rt_vals
            self.logprobs[sidx:eidx]=logprobs
            self.index=eidx
            
            # clean temporary buffer
            self.t_buffer=[]

    def size(self):
        return self.n

    def reset(self):
        self.index=0
        self.n=0
        
    def get_all(self):
        assert self.n==self.index
        return self.states[:self.index],self.actions[:self.index],self.rt_vals[:self.index],self.adv_vals[:self.index],self.logprobs[:self.index]
    
    def get_batch(self,n):
        assert n<=self.size(),"cannot exceed the size of buffer!"
        if n==self.size():
            return self.get_all()
        else:
            inds=np.random.choice(self.size(),n,replace=False)
            return self.states[inds],self.actions[inds],self.rt_vals[inds],self.adv_vals[inds],self.logprobs[inds]