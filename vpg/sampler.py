class Sampler:
    def __init__(self,params):
        self.params=params
        self.agent=params["agent"]
        self.buffer=params["buffer"]
        self.metrics=params["metrics"]
        self.env=params["env"]
        self.gamma=params["gamma"]
        self.o=self.env.reset()
        self.rt_val=0
        self.trajectory_len=0
        self.discount=1
        
    def sample_batch(self):
        #todo bug too small size
        self.buffer.reset()
        while self.buffer.size()<self.params["batch_size_pc"]:
            # sample action from agent
            a,logprob=self.agent.act(self.o)
            n_o,r,terminal,_=self.env.step(a)
            self.trajectory_len+=1
            self.rt_val+=r*self.discount
            self.discount*=self.gamma
            self.buffer.append(self.o,a,r,terminal,logprob)
            self.o=n_o
            if terminal:
                self.metrics.add("trajectory_len",self.trajectory_len)
                self.metrics.add("rt_val",self.rt_val)
                self.rt_val=0
                self.trajectory_len=0
                self.discount=1
                self.o=self.env.reset()
            
  