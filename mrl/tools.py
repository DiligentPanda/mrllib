import scipy.signal
import numpy as np
import torch
from mpi_tools import proc_id

def get_gpu(gpus):
    idx=proc_id()%len(gpus)
    gpu=gpus[idx]
    device=torch.device(f"cuda:{gpu}")
    return device

def to_tensor(arr):
    if isinstance(arr,np.ndarray):
        if arr.dtype==np.float64:
            arr=arr.astype(np.float32)
        return torch.from_numpy(arr),np.ndarray
    elif isinstance(arr,torch.Tensor):
        arr=arr.float()
        return arr,torch.Tensor
    else:
        raise NotImplementedError

def to_tensors(*data):
    return [to_tensor(d)[0] for d in data]
    
def to_array(tensor:torch.Tensor):
    if isinstance(tensor,np.ndarray):
        return tensor,np.ndarray
    elif isinstance(tensor,torch.Tensor):
        return tensor.numpy(),torch.Tensor
    else:
        raise NotImplementedError

def to_arrays(*data):
    return [to_array(d)[0] for d in data]

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]