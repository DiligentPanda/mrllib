from distutils.command.config import config
import gym
import torch
import numpy as np
from agent import VPGAgent, VPGBuffer
from processor import InputProcessor
from torch.distributions import Distribution
from torch.optim import Adam
import torch.nn as nn
from logger import Logger
from metric import Metrics,Timer
from config import Config
from sampler import Sampler
from mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import random


class Trainer:
    def __init__(self,params):
        setup_pytorch_for_mpi()
        self.params=params
        self.seed()
        self.logger=Logger("trainer",params["log_fp"])
        self.batch_size=params["batch_size"]
        self.env_name=params["env_name"]
        self.env=gym.make(self.env_name)
        self.agent:nn.Module=VPGAgent(params)
        sync_params(self.agent.actor)
        self.buffer=VPGBuffer(params)
        self.input_processor=InputProcessor()
        self.lr=params["lr"]
        self.actor_optim=Adam(self.agent.actor.parameters(),self.lr)
        self.metrics=Metrics(
            names=[
                "rt_val",
                "trajectory_len",
                "train_time",
                "sample_time",
                "learn_time",
            ],
            capacity=self.params["metric_capacity"]
        )
        self.timer=Timer()

        self.params["agent"]=self.agent
        self.params["buffer"]=self.buffer
        self.params["metrics"]=self.metrics
        self.params["env"]=self.env
        self.sampler=Sampler(params)
        
    def seed(self):
        seed=self.params["rnd_seed"] + 10000 * proc_id()
        # random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train_step(self):
        self.timer.record("step_s")
        # sample data from environments
        self.sampler.sample_batch()
        self.metrics.add("sample_time",self.timer.time("step_s","sample_e"))
        # optimize 
        ## get batch of data from replay buffer
        data=self.buffer.get_data()
        states,actions,rt_vals=self.input_processor.process(data)
        ## calculate loss
        dists:Distribution=self.agent.actor(states)
        loss=torch.mean(-dists.log_prob(actions)*rt_vals)
        ## zero gradients
        self.actor_optim.zero_grad()
        ## backward
        loss.backward()
        mpi_avg_grads(self.agent.actor)
        ## step optimizer
        self.actor_optim.step()
        self.metrics.add("learn_time",self.timer.time("sample_e"))
        self.metrics.add("train_time",self.timer.time("step_s"))
        if self.iter%self.params["log_freq"]==0:
            # max_val=self.metrics.max("rt_val")
            mean_val=self.metrics.mean("rt_val")
            # min_val=self.metrics.min("rt_val")
            # max_len=self.metrics.max("trajectory_len")
            # mean_len=self.metrics.mean("trajectory_len")
            # min_len=self.metrics.min("trajectory_len")
            # train_time=self.metrics.mean("train_time")
            # sample_time=self.metrics.mean("sample_time")
            # learn_time=self.metrics.mean("learn_time")
            
            self.logger.info(f"<ITER {self.iter} {mean_val}>")
                            #  f"val(max,mean,min):({max_val:.2f},{mean_val:.2f},{min_val:.2f})    "
                            #  f"len(max,mean,min):({max_len:.2f},{mean_len:.2f},{min_len:.2f})    "
                            #  f"time(train,learn,sample):({train_time:.2f},{learn_time:.2f},{sample_time:.2f})")
        
    def train(self):
        s_iter=1
        e_iter=self.params["total_iters"]+1
        
        if self.params["load_ckpt"]:
            obj=self.load_checkpoint(self.params["load_ckpt"])
            s_iter=obj["iter"]+1
            self.agent.actor.load_state_dict(obj["actor"])
            self.actor_optim.load_state_dict(obj["actor_optim"])
        
        self.logger.info("Start Training...")
        for self.iter in range(s_iter,e_iter):
            self.train_step()
            if self.iter%self.params["ckpt_freq"]==0:
                obj={
                    "iter":iter,
                    "actor":self.agent.actor.state_dict(),
                    "actor_optim":self.actor_optim.state_dict()
                }
            
            
    def save_checkpoint(self,obj,fp):
        torch.save(obj,fp)
    
    def load_checkpoint(self,fp):
        return torch.load(fp)
    
if __name__=="__main__":
    config=Config()
    config.build()
    mpi_fork(config["n_cpus"])
    if proc_id()==0:
        config.dump()
    trainer=Trainer(config)
    trainer.train()