from critic import MLPCritic
from actor import MLPActor
import numpy as np
from tools import to_tensor,to_arrays
import torch.nn as nn

class VPGAgent(nn.Module):
    '''
    REINFORCE
    '''
    def __init__(self,params):
        nn.Module.__init__(self)
        self.actor=MLPActor(params)
        
    def forward(self):
        pass
    
    def act(self,obs):
        obs,old_type=to_tensor(obs)
        actions,logprobs=self.actor.sample_action(obs)
        if old_type==np.ndarray:
            actions,logprobs=to_arrays(actions,logprobs)
        return actions,logprobs