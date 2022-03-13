from distutils.command.config import config
import gym
import torch
import numpy as np
from agents.vpg_agent import VPGAgent
from agents.ppo_agent import PPOAgent
from processor import InputProcessor
import torch.nn as nn
from logger import Logger
from metric import Metrics,Timer
from config import Config
from sampler import Sampler
from mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import random
from buffer import VPGBuffer
import torch.nn.functional as F
import pprint

class Trainer:
    def __init__(self,params):
        setup_pytorch_for_mpi()
        self.name=self.__class__.__name__
        self.params=params
        self.logger=Logger(self.name,params["log_fp"])
        self.logger.info(pprint.pformat(self.params))
        self.metrics=Metrics(
            names=[
                "rt_val",
                "trajectory_len",
                "train_time",
                "sample_time",
                "learn_time",
                "val_start",
                "sample_steps"
            ],
            capacities=[10,10,1,1,1,10,1]
        )
        self.params["metrics"]=self.metrics
        self.timer=Timer()
        self.batch_size=params["batch_size"]
        self.env_name=params["env_name"]
        self.env=gym.make(self.env_name)
        self.seed()
        self.params["action_space"]=self.env.action_space
        self.params["observation_space"]=self.env.observation_space
        self.buffer=VPGBuffer(params)
        self.params["buffer"]=self.buffer
        self.input_processor=InputProcessor(params)
        self.params["input_processor"]=self.input_processor
        if self.params["algo"]=="vpg":
            self.agent:nn.Module=VPGAgent(params)
        elif self.params["algo"]=="ppo":
            self.agent:nn.Module=PPOAgent(params)
        else:
            raise NotImplementedError
        sync_params(self.agent.actor)
        self.params["agent"]=self.agent
        self.params["metrics"]=self.metrics
        self.params["env"]=self.env
        self.sampler=Sampler(params)
        
    def seed(self):
        seed=self.params["rnd_seed"] + 10000 * proc_id()
        # random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.env.seed(seed)

    def train_step(self):
        self.timer.record("step_s")
        # sample data from environments
        self.sampler.sample_batch()
        self.metrics.sample_time.update(self.timer.time("step_s","sample_e"))
        # learn from data
        ret=self.agent.learn()
        self.metrics.learn_time.update(self.timer.time("sample_e"))
        self.metrics.train_time.update(self.timer.time("step_s"))
        self.agent.report(self.iter,ret)
        
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
                pass
                # obj={
                #     "iter":iter,
                #     "actor":self.agent.actor.state_dict(),
                #     "actor_optim":self.agent.actor_optim.state_dict()
                # }
                # todo save checkpoint
            
            
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