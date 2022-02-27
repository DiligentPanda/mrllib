import torch.nn as nn
import torch
from net import mlp

class MLPCritic(nn.Module):
    def __init__(self,params) -> None:
        super().__init__()
        self.params=params
        self.sizes=self.params["mlp_sizes"]
        # could be set by env
        self.obs_dim=self.params["obs_ndim"]
        self.n_values=self.params["n_values"]
        self.mlp=mlp(self.sizes,nn.ReLU)
        self.head=nn.Linear(self.sizes[-1],self.n_values)
        
    def forward(self,obs):
        feats=self.mlp(obs)
        vals=self.head(feats)
        return vals
    
    def evaluate(self,obs):
        old_dim=obs.dim()
        if old_dim==self.obs_dim:
            obs=obs.unsqueeze(0)
        else:
            assert obs.dim==self.obs_dim+1
        with torch.no_grad():
            vals=self.forward(obs)
        if old_dim!=vals.dim():
            vals=vals.squeeze(0)
        return vals