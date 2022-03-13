import torch.nn as nn
import torch
from net import mlp
from torch.distributions import Categorical,Normal,Distribution
import numpy as np
from logger import Logger
from gym.spaces import Box,Discrete

class MLPActor(nn.Module):
    def __init__(self,params) -> None:
        super().__init__()
        self.params=params
        self.obs_shape=self.params["observation_space"].shape
        self.obs_dim=len(self.obs_shape)
        assert self.obs_dim==1
        self.sizes=list(self.obs_shape)+self.params["mlp_sizes"]
        
        action_space=self.params["action_space"]
        if isinstance(action_space,Discrete):
            self.discrete=True
            self.n_actions=action_space.n
        elif isinstance(action_space,Box):
            self.discrete=False
            self.n_actions=action_space.shape[0]
            # std
            self.log_std=nn.Parameter(self.params["log_std"]*torch.ones(self.n_actions,dtype=torch.float32))
            
        self.mlp=mlp(self.sizes,nn.Tanh)
        self.head=nn.Linear(self.sizes[-1],self.n_actions)
        self.logger=Logger("actor",self.params["log_fp"])
        
    def forward(self,obs)->Distribution:
        feats=self.mlp(obs)
        if self.discrete:
            logits=self.head(feats)
            dists=Categorical(logits=logits)
        else:
            mu=self.head(feats)
            std=torch.exp(self.log_std)
            dists=Normal(mu,std)
        return dists
    
    def get_log_prob(self,dists,actions):
        log_probs=dists.log_prob(actions)
        if not self.discrete:
            log_probs=log_probs.sum(axis=-1)
        return log_probs
        
    def sample_action(self,obs):
        unsqueezed=False
        if obs.dim()==self.obs_dim:
            obs=obs.unsqueeze(0)
            unsqueezed=True
        else:
            assert obs.dim()==self.obs_dim+1
        with torch.no_grad():
            dists=self.forward(obs)
            actions=dists.sample()
            logprobs=dists.log_prob(actions)
            assert logprobs.shape==actions.shape
        
        if unsqueezed:
            # single sample
            actions=actions.squeeze(0)
            logprobs=logprobs.squeeze(0)
        if not self.discrete:
            logprobs=logprobs.sum(axis=-1)
        return actions,logprobs