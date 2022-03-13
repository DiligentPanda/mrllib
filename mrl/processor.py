from tools import to_tensors
from mpi_tools import mpi_statistics_scalar
from gym.spaces import Box,Discrete

class InputProcessor:
    def __init__(self,params):
        self.params=params
        
    def process(self,data):
        states,actions,rt_vals,adv_vals,logprobs=data
        # normalization
        mean,std=mpi_statistics_scalar(adv_vals)
        if self.params["normalize_mean"]:
            adv_vals-=mean
        if self.params["normalize_std"]:
            assert self.params["normalize_mean"]
            adv_vals/=(std+1e-20)
        # convert to tensors
        states,actions,rt_vals,adv_vals,logprobs=to_tensors(states,actions,rt_vals,adv_vals,logprobs)
        # TODO: devices 
        states=states.float()
        if isinstance(self.params["action_space"],Discrete):
            actions=actions.long() # this only works in discrete case
        elif isinstance(self.params["action_space"],Box):
            actions=actions.float()
        rt_vals=rt_vals.float()
        adv_vals=adv_vals.float()
        logprobs=logprobs.float()

        return states,actions,rt_vals,adv_vals,logprobs