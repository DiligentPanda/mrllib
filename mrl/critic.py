import torch.nn as nn
import torch
from net import mlp

class MLPCritic(nn.Module):
    def __init__(self,params) -> None:
        super().__init__()
        self.params=params
        self.obs_shape=self.params["observation_space"].shape
        self.obs_dim=len(self.obs_shape)
        assert self.obs_dim==1
        self.sizes=list(self.obs_shape)+self.params["mlp_sizes"]
        # could be set by env
        self.n_values=self.params["n_values"]
        self.mlp=mlp(self.sizes,nn.Tanh)
        self.head=nn.Linear(self.sizes[-1],self.n_values)
        
    def forward(self,obs):
        feats=self.mlp(obs)
        vals=self.head(feats)
        if self.n_values==1:
            vals=vals.squeeze(-1)
        return vals
    
    def evaluate(self,obs):
        unsqueezed=False
        if obs.dim()==self.obs_dim:
            obs=obs.unsqueeze(0)
            unsqueezed=True
        else:
            assert obs.dim()==self.obs_dim+1
        with torch.no_grad():
            vals=self.forward(obs)
        if unsqueezed:
            vals=vals.squeeze(0)
        return vals