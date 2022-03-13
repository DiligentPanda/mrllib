from critic import MLPCritic
from actor import MLPActor
from critic import MLPCritic
import numpy as np
from tools import to_tensor,to_array,to_arrays
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Distribution
import torch
from mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import torch.nn.functional as F
from logger import Logger

class VPGAgent(nn.Module):
    '''
    REINFORCE
    '''
    def __init__(self,params):
        nn.Module.__init__(self)
        self.params=params
        self.logger=Logger(self.params["algo"],params["log_fp"])
        self.actor=MLPActor(params)
        self.critic=MLPCritic(params)
        self.actor_optim=Adam(self.actor.parameters(),self.params["plr"])
        self.critic_optim=Adam(self.critic.parameters(),self.params["vlr"])
        self.metrics=params["metrics"]
        self.buffer=params["buffer"]
        self.input_processor=params["input_processor"]
        
    def forward(self):
        pass
    
    def act(self,obs):
        obs,old_type=to_tensor(obs)
        actions,logprobs=self.actor.sample_action(obs)
        if old_type==np.ndarray:
            actions,logprobs=to_arrays(actions,logprobs)
        return actions,logprobs
    
    def eval(self,obs):
        obs,old_type=to_tensor(obs)
        values=self.critic.evaluate(obs)
        if old_type==np.ndarray:
            values,_=to_array(values)
        return values
    
    def learn(self):
        ret={}
        old_logprobs=None
        old_ploss=None
        ## get batch of data from replay buffer
        data=self.buffer.get_all()
        data=self.input_processor.process(data)
        states,actions,rt_vals,adv_vals,_=data
        ## update
        ### update policy
        dists:Distribution=self.actor(states)
        logprobs=self.actor.get_log_prob(dists,actions)
        ploss=torch.mean(-logprobs*adv_vals)
        if old_logprobs is None:
            old_logprobs=logprobs
            old_ploss=ploss
        #### zero gradients
        self.actor_optim.zero_grad()
        #### backward
        ploss.backward()
        mpi_avg_grads(self.actor)
        #### step optimizer
        self.actor_optim.step()
        
        #### todo for debug? if delta loss increases then lr is too large?
        #### kl divergence should be small.
        with torch.no_grad():
            new_dists=self.actor(states)
            new_logprobs=self.actor.get_log_prob(new_dists,actions)
            new_ploss=torch.mean(-new_logprobs*adv_vals)
            delta_ploss=new_ploss.item()-old_ploss.item() 
            kl_div=(old_logprobs-new_logprobs).mean().item()
            entropy=new_dists.entropy().mean().item()
        ploss=ploss.item()
        ret.update({
            "entropy":entropy,
            "delta_ploss":delta_ploss,
            "kl_div":kl_div,
            "ploss":ploss
        })
        
        if self.params["use_adv"]:
            ### update value function
            vloss_init=None
            vloss_last=None
            dec_iter=0
            for iter in range(self.params["train_v_iters"]):
                ## get batch of data from replay buffer
                values=self.critic(states)
                # self.logger.debug(f"rt_vals:{rt_vals}")
                # self.logger.debug(f"values:{values}")
                vloss=F.mse_loss(values,rt_vals)
                vloss_n=vloss.item()
                if vloss_init is None:
                    vloss_init=vloss_n
                else:
                    vloss_diff=vloss_n-vloss_last
                    dec_iter+=1 if vloss_diff<0 else 0
                vloss_last=vloss_n
                self.critic_optim.zero_grad()
                vloss.backward()
                mpi_avg_grads(self.critic)
                self.critic_optim.step()
            # for debug
            with torch.no_grad():
                values=self.critic(states)
                vloss=F.mse_loss(values,rt_vals)
                vloss_n=vloss.item()
                vloss_diff=vloss_n-vloss_last
                dec_iter+=1 if vloss_diff<0 else 0
                vloss_diff=vloss_n-vloss_init
                dec_rate=dec_iter/self.params["train_v_iters"]*100
            
            ret.update({
                "vloss":vloss_n,
                "vloss_init":vloss_init,
                "vloss_diff":vloss_diff,
                "dec_rate":dec_rate
            })
            
            return ret
        
    def report(self,iter,ret):
        entropy=ret["entropy"]
        kl_div=ret["kl_div"]
        delta_ploss=ret["delta_ploss"]
        ploss=ret["ploss"]
        if self.params["use_adv"]:
            vloss=ret["vloss"]
            vloss_init=ret["vloss_init"]
            vloss_diff=ret["vloss_diff"]
            dec_rate=ret["dec_rate"]
            
        if iter%self.params["log_freq"]==0:
            val_start=self.metrics.val_start.value
            mean_val,std_val,min_val,max_val=self.metrics.rt_val.rstat(True)
            mean_len,std_len,min_len,max_len=self.metrics.trajectory_len.rstat(True)
            train_time=self.metrics.train_time.mean
            sample_time=self.metrics.sample_time.mean
            learn_time=self.metrics.learn_time.mean
            sample_steps=self.metrics.sample_steps.sum
            
            self.logger.info(f"<ITER {iter}> Sample Steps:{sample_steps}\n"
                             f"val(mean,std,max,min):({mean_val:.2f},{std_val:.2f},{max_val:.2f},{min_val:.2f})\n"
                             f"len(mean,std,max,min):({mean_len:.2f},{std_len:.2f},{max_len:.2f},{min_len:.2f})\n"
                             f"ploss:{ploss:.6e} entropy:{entropy:.6e} kl_div:{kl_div:.6e} delta_ploss:{delta_ploss:.6e}\n"+
                             (f"val_start:{val_start:.2f} vloss:{vloss:.6e} vloss_init:{vloss_init:.6e} vloss_diff:{vloss_diff:.6e} dec_rate:{dec_rate:.2f}%\n"
                             if self.params["use_adv"] else "")
                             +f"time(train,learn,sample):({train_time:.2f},{learn_time:.2f},{sample_time:.2f})")