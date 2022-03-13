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
from .vpg_agent import VPGAgent

class PPOAgent(VPGAgent):
    '''
    Reinforce with clipped reward, allowing for multiple steps of update from a single batch of samples.
    '''
    def __init__(self,params):
        VPGAgent.__init__(self,params)
        self.upper_ratio=self.params["upper_ratio"]
        self.lower_ratio=self.params["lower_ratio"]
        
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
        
        for p_iter in range(self.params["train_p_iters"]):
            ## get batch of data from replay buffer
            data=self.buffer.get_batch(n=self.params["train_p_batch_pc"])
            data=self.input_processor.process(data)
            states,actions,rt_vals,adv_vals,old_logprobs_=data
            ## update
            ### update policy
            dists:Distribution=self.actor(states)
            logprobs=self.actor.get_log_prob(dists,actions)
            ratios=torch.exp(logprobs-old_logprobs_)
            ploss=-torch.mean(
                torch.min(
                    ratios*adv_vals,
                    torch.clamp(ratios,self.lower_ratio,self.upper_ratio)*adv_vals
                )
            )
            if self.params["train_p_early_stop"]:
               kl=(old_logprobs_-logprobs).mean().item()
               kl = mpi_avg(kl)
               if kl > self.params["target_kl"]:
                   self.logger.info(f'Early stopping at step {p_iter} due to reaching max kl.')
                   break
            #### zero gradients
            self.actor_optim.zero_grad()
            #### backward
            ploss.backward()
            mpi_avg_grads(self.actor)
            #### step optimizer
            self.actor_optim.step()
        
        
        data=self.buffer.get_all()
        data=self.input_processor.process(data)
        states,actions,rt_vals,adv_vals,old_logprobs=data
        
        ### todo for debug? if delta loss increases then lr is too large?
        ### kl divergence should be small.
        with torch.no_grad():
            dists=self.actor(states)
            logprobs=self.actor.get_log_prob(dists,actions)
            ratios=torch.exp(logprobs-old_logprobs)
            clipped=ratios.gt(self.upper_ratio)|ratios.lt(self.lower_ratio)
            clip_frac=clipped.float().mean().item()*100
            ploss=-torch.mean(adv_vals)
            new_ploss=-torch.mean(ratios*adv_vals)
            delta_ploss=new_ploss.item()-ploss.item() 
            kl_div=(old_logprobs-logprobs).mean().item()
            entropy=dists.entropy().mean().item()
            ploss=ploss.item()
 
        ret.update({
            "entropy":entropy,
            "delta_ploss":delta_ploss,
            "kl_div":kl_div,
            "ploss":ploss,
            "clip_frac":clip_frac
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
        clip_frac=ret["clip_frac"]
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
                             f"ploss:{ploss:.6e} entropy:{entropy:.6e} kl_div:{kl_div:.6e} delta_ploss:{delta_ploss:.6e} clip_frac:{clip_frac:.2}%\n"+
                             (f"val_start:{val_start:.2f} vloss:{vloss:.6e} vloss_init:{vloss_init:.6e} vloss_diff:{vloss_diff:.6e} dec_rate:{dec_rate:.2f}%\n"
                             if self.params["use_adv"] else "")
                             +f"time(train,learn,sample):({train_time:.2f},{learn_time:.2f},{sample_time:.2f})")