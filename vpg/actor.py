import torch.nn as nn
import torch
from net import mlp
from torch.distributions import Categorical
import numpy as np
from logger import Logger

class MLPActor(nn.Module):
    def __init__(self,params) -> None:
        super().__init__()
        self.params=params
        self.state_shape=self.params["state_shape"]
        assert len(self.state_shape)==1
        self.sizes=list(self.state_shape)+self.params["mlp_sizes"]
        # could be set by env
        self.obs_dim=self.params["obs_ndim"]
        self.n_actions=self.params["n_actions"]
        self.mlp=mlp(self.sizes,nn.ReLU)
        self.head=nn.Linear(self.sizes[-1],self.n_actions)
        self.logger=Logger("actor",self.params["log_fp"])
        
    def forward(self,obs)->Categorical:
        feats=self.mlp(obs)
        logits=self.head(feats)
        dists=Categorical(logits=logits)
        return dists
        
    def sample_action(self,obs):
        old_dim=obs.dim()
        if old_dim==self.obs_dim:
            obs=obs.unsqueeze(0)
        else:
            assert obs.dim==self.obs_dim+1
        with torch.no_grad():
            dists=self.forward(obs)
            actions=dists.sample(torch.Size([1]))
            actions=actions.squeeze(-1)
            logprobs=dists.log_prob(actions)
            assert logprobs.shape==actions.shape
            
        #self.logger.debug("obs_shape: {},actions_shape:{}".format(obs.shape,actions.shape))
        
        if old_dim!=obs.dim():
            # single sample
            actions=actions.squeeze(0)
            logprobs=logprobs.squeeze(0)
        return actions,logprobs