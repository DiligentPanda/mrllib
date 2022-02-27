from tools import to_tensors
from mpi_tools import mpi_statistics_scalar

class InputProcessor:
    def __init__(self,params):
        self.params=params
        
    def process(self,data):
        states,actions,rt_vals,logprobs=data
        # normalization
        mean,std=mpi_statistics_scalar(rt_vals)
        if self.params["normalize_mean"]:
            rt_vals-=mean
        if self.params["normalize_std"]:
            assert self.params["normalize_mean"]
            rt_vals/=(std+1e-20)
        # convert to tensors
        states,actions,rt_vals,logprobs=to_tensors(states,actions,rt_vals,logprobs)
        # TODO: devices 
        # types
        states=states.float()
        actions=actions.long()
        rt_vals=rt_vals.float()
        logprobs=logprobs.float()

        return states,actions,rt_vals,logprobs