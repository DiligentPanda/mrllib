class Sampler:
    def __init__(self,params):
        self.params=params
        self.agent=params["agent"]
        self.buffer=params["buffer"]
        self.metrics=params["metrics"]
        self.env=params["env"]
        self.o=self.env.reset()
        self.rt_val=0
        self.trajectory_len=0
        
    def sample_batch(self):
        #todo bug too small size
        self.buffer.reset()
        while self.buffer.size()<self.params["batch_size_pc"]:
            # sample action from agent
            a=self.agent.act(self.o)
            n_o,r,terminal,_=self.env.step(a)
            self.trajectory_len+=1
            self.rt_val+=r
            self.buffer.append(self.o,a,r,terminal)
            self.o=n_o
            if terminal:
                self.metrics.add("trajectory_len",self.trajectory_len)
                self.metrics.add("rt_val",self.rt_val)
                self.rt_val=0
                self.trajectory_len=0
                self.o=self.env.reset()
            
  