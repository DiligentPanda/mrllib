from logger import Logger
import numpy as np

class Sampler:
    def __init__(self,params):
        self.params=params
        self.agent=params["agent"]
        self.buffer=params["buffer"]
        self.metrics=params["metrics"]
        self.env=params["env"]
        self.gamma=params["gamma"]
        self.max_ep_len=params["max_ep_len"]
        self.o=self.env.reset()
        self.rt_val=0
        self.trajectory_len=0
        self.v=None
        self.logger=Logger("sampler",params["log_fp"])
        
    def sample_batch(self):
        batch_size=self.params["batch_size_pc"]
        #todo bug too small size
        self.buffer.reset()
        while self.buffer.size()<batch_size:
            # sample action from agent
            a,logprob=self.agent.act(self.o)
            v=self.agent.eval(self.o)
            if self.v is None:
                self.v=v
            # todo 
            n_o,r,terminal,_=self.env.step(a)
            self.trajectory_len+=1
            self.rt_val+=r
            self.buffer.append(self.o,a,r,v,logprob)
            self.o=n_o
            if terminal or self.trajectory_len==self.max_ep_len or self.buffer.size()==batch_size:
                if terminal:
                    last_val=0
                else:
                    last_val=self.agent.eval(self.o)
                    # todo warning: cut off
                self.buffer.finish(last_val)
                if terminal or self.trajectory_len==self.max_ep_len:
                    self.metrics.trajectory_len.update(self.trajectory_len)
                    self.metrics.rt_val.update(self.rt_val)
                    self.metrics.val_start.update(self.v)
                self.metrics.sample_steps.update(self.trajectory_len)
                self.v=None
                self.rt_val=0
                self.trajectory_len=0
                self.o=self.env.reset()
                if self.buffer.size()==batch_size:
                    break
        
  