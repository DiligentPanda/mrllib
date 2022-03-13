import yaml
import argparse
import os
import time

class Config:
    def __init__(self) -> None:
        self.params={}
    
    def build(self):
        parser=argparse.ArgumentParser()
        parser.add_argument('--cfg',type=str,default="configs/basic.yaml")
        parser.add_argument('--exp_fd',type=str)
        parser.add_argument('--n_cpus',type=int)
        args=parser.parse_args()
        
        self.load_yaml(args.cfg)
        self.add_args(args)
        exp_fd=self.params["exp_fd"]
        os.makedirs(exp_fd,exist_ok=True)
        self.timestamp=time.strftime("%y-%m-%d-%H-%M-%S",time.localtime())
        self.params["log_fp"]=os.path.join(exp_fd,f"log_{self.timestamp}.txt")
        assert self.params["batch_size"]%self.params["n_cpus"]==0
        self.params["batch_size_pc"]=int(self.params["batch_size"]/self.params["n_cpus"])
        self.params["train_p_batch_pc"]=int(self.params["train_p_batch"]/self.params["n_cpus"])
        self.params["n_gpus"]=len(self.params["gpus"])
    
    def dump(self):
        # dump config
        with open(os.path.join(self.params["exp_fd"],f"config_{self.timestamp}.yaml"),'w') as f:
            yaml.dump(self.params,f)
        
    def load_yaml(self,fp):
        with open(fp) as f:
            params=yaml.safe_load(f)
        self.params.update(params)
    
    def add_args(self,args):
        for k,v in vars(args).items():
            if v is not None:
                self.params[k]=v
    
    def __getitem__(self,key):
        return self.params[key]
    
    def __setitem__(self,key,value):
        self.params[key]=value