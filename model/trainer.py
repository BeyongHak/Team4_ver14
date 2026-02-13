from torch.distributions import Normal,kl_divergence
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
import os
import os
import numpy as np
import datetime
from model.model import PPOMLPModel
from model.agent import Agent
from model.writer import Writer
from model.distribution import Distribution

class Trainer:
    """Train the model"""
    def __init__(self,config,env,writer_path=None,save_path=None, max_updates=1000000) -> None:
        self.config        = config
        self.env           = env
        
        # [New] Logging Info
        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.config_name = config.get("config_name", "UnknownConfig")

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')


        self.model         = PPOMLPModel(config,self.env.getStateSize(),self.env.getActionSize())
        self.model.to(self.device)
        
        # [Mod] Dual Optimizers
        # Group Parameters
        # Actor: MLP_A (Embedding) + Policy Head
        actor_params = list(self.model.mlp_A.parameters()) + list(self.model.policy.parameters())
        
        # Critic: MLP_C (Embedding) + Value Head
        critic_params = list(self.model.mlp_C.parameters()) + list(self.model.value.parameters())
        
        # Note: If there are shared parameters, they need handling, but current PPOMLPModel has distinct paths.
        # mlp_A is for retailer state (used by Actor)
        # mlp_C is for global state (used by Critic)
        # So they are fully disjoint.
        
        self.optimizer_actor = torch.optim.AdamW(actor_params, lr=config['lr'])
        self.optimizer_critic = torch.optim.AdamW(critic_params, lr=config['lr'])
        
        if writer_path is not None:
            self.writer    = Writer(writer_path)
        if save_path is not None:
            self.save_path = save_path

        self.agent         = Agent(self.env,self.model,config, self.device)
        self.dist          = Distribution()

        try:
            self.model.load_state_dict(torch.load(f'{save_path}model.pt'))
            with open(f"{save_path}stat.json","r") as f:
                self.data = json.load(f)
            print('PROGRESS RESTORED!')
        except:
            print("TRAIN FROM BEGINING!")
            self.data = {
                "step":0,
                "entropy_coef":config["entropy_coef"]["start"]
            }

        self.entropy_coef = self.data["entropy_coef"]
        self.entropy_coef = self.data["entropy_coef"]
        
        # [Mod] Entropy Schedule adjusted for Critic Warmup
        # Decay should happen from [Warmup End] to [Config Step End]
        # e.g. 10,000 to 100,000 -> Duration = 90,000
        self.critic_warmup_steps = config.get("critic_warmup_steps", 10000)
        total_decay_steps = config['entropy_coef']['step']
        
        effective_decay_duration = max(1, total_decay_steps - self.critic_warmup_steps)
        
        self.entropy_coef_step = (config["entropy_coef"]["start"] - config['entropy_coef']['end']) / effective_decay_duration
        
        self.total_env_steps = 0 
        self.max_updates = max_updates    
        self.warmup_updates = 100000 

    def _entropy_coef_schedule(self):
        self.entropy_coef -= self.entropy_coef_step
        if self.entropy_coef <= self.config['entropy_coef']['end']:
            self.entropy_coef = self.config['entropy_coef']['end']

    def _calculate_mono_loss(self, states):
        """
        Calculate Diagonal Monotonicity Loss for IMMEDIATE neighbors only (Step +/- 1).
        
        Path:
        - Lower: S_curr - 1 (All retailers -1)
        - Upper: S_curr + 1 (All retailers +1)
        
        Constraints:
        - Lower Path: Action(S_lower) <= Action(S_curr). Loss if >. Gradient on S_curr.
        - Upper Path: Action(S_curr) <= Action(S_upper). Loss if >. Gradient on S_upper.
        """
        batch_size = states.shape[0]
        device = states.device
        
        # 1. Setup & Dimensions
        inv_input = states[:, :self.model.inv_dim] # (B, Inv)
        ret_input = states[:, self.model.inv_dim:] # (B, N)
        
        max_k_tensor = self.model.retailer_max_k_tensor.to(device) # (N,)
        
        # Current Integer State
        k_int = torch.round(ret_input * max_k_tensor) # (B, N)
        
        # 2. Define Valid Neighbors (Step = 1)
        
        # Lower Valid: All k_i >= 1
        # min(k_int) >= 1
        valid_lower = (k_int.min(dim=1).values >= 1.0) # (B,)
        
        # Upper Valid: All k_i < Max K_i
        # We check dist_to_max >= 1
        dist_to_max = max_k_tensor - k_int
        valid_upper = (dist_to_max.min(dim=1).values >= 1.0) # (B,)
        
        # 3. Create Neighbor States
        # Step size in normalized space = 1.0 / MaxK
        delta = 1.0 / max_k_tensor # (N,) 
        
        loss_lower = torch.tensor(0.0, device=device)
        loss_upper = torch.tensor(0.0, device=device)
        
        # --- Lower Check (S_prev vs S_curr) ---
        if valid_lower.any():
            # Create S_prev
            ret_lower = ret_input - delta
            ret_lower = torch.clamp(ret_lower, min=0.0)
            
            # Filter only valid samples
            ret_lower_valid = ret_lower[valid_lower]
            inv_lower_valid = inv_input[valid_lower]
            
            states_lower = torch.cat([inv_lower_valid, ret_lower_valid], dim=1)
            
            # Action(Lower) - No Grad (Anchor)
            with torch.no_grad():
                saa_lower = self.model.get_saa_base_action(states_lower)
                (mean_lower, _), _ = self.model(states_lower)
                act_lower = F.relu(saa_lower - mean_lower)
            
            # Action(Curr) - With Grad (Target to change)
            states_curr_valid = states[valid_lower]
            saa_curr = self.model.get_saa_base_action(states_curr_valid)
            (mean_curr, _), _ = self.model(states_curr_valid)
            act_curr = F.relu(saa_curr - mean_curr) 
            
            # Constraint: Lower <= Curr
            # Violation: Lower > Curr
            # Loss = ReLU(Lower - Curr)
            diff_lower = act_lower - act_curr
            loss_vec = F.relu(diff_lower)
            
            # Normalize by Batch Size (not valid count) for stability
            loss_lower = loss_vec.sum() / batch_size
            
        # --- Upper Check (S_curr vs S_next) ---
        if valid_upper.any():
            # Create S_next
            ret_upper = ret_input + delta
            
            ret_upper_valid = ret_upper[valid_upper]
            inv_upper_valid = inv_input[valid_upper]
            
            states_upper = torch.cat([inv_upper_valid, ret_upper_valid], dim=1)
            
            # Action(Curr) - No Grad (Anchor)
            states_curr_valid = states[valid_upper]
            with torch.no_grad():
                saa_curr = self.model.get_saa_base_action(states_curr_valid)
                (mean_curr, _), _ = self.model(states_curr_valid)
                act_curr = F.relu(saa_curr - mean_curr)
                
            # Action(Upper) - With Grad (Target to change/pull up)
            saa_upper = self.model.get_saa_base_action(states_upper)
            (mean_upper, _), _ = self.model(states_upper)
            act_upper = F.relu(saa_upper - mean_upper)
            
            # Constraint: Curr <= Upper
            # Violation: Curr > Upper
            # Loss = ReLU(Curr - Upper)
            diff_upper = act_curr - act_upper
            loss_vec = F.relu(diff_upper)
            
            loss_upper = loss_vec.sum() / batch_size

        return loss_lower, loss_upper

    def _standard_ppo_loss(self,value,value_new,entropy,log_prob,log_prob_new,advantage,returns):
        """
        Overview:
            Calculates the total loss using Standard PPO method.
        """
        
        if self.config["normalize_advantage"]:
            advantage   = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
        # PPO Ratio
        ratios = torch.exp(torch.clamp(log_prob_new - log_prob, min=-20., max=5.))
        
        # Policy Loss
        surr1 = ratios * advantage
        eps_clip = self.config["PPO"].get("policy_clip", 0.2) # Default 0.2 if not present
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value Loss (Clipped)
        value_clipped = value + torch.clamp(value_new - value, -self.config["PPO"]["value_clip"], self.config["PPO"]["value_clip"])
        value_losses = (returns - value_new) ** 2
        value_losses_clipped = (returns - value_clipped) ** 2
        critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
  
        # Total Loss is still needed for backward, but we can verify gradients flow correctly
        total_loss = actor_loss + self.config["PPO"]["critic_coef"] * critic_loss - self.entropy_coef * entropy.mean()

        return actor_loss, critic_loss, total_loss, entropy.mean()
    
    def train(self,write_data=True, result_callback=None):
        """
        Overview:
            Trains the model.

        Arguments:
            - write_data: (`bool`): Whether to write data to Tensorboard or not.
            - result_callback: (`callable`): Function to call every step (returns True to stop).
        """
        training = True

        while training:

            Avg_reward = self.agent.run(num_games=self.config["num_game_per_batch"])
            
            # [New] Track total steps
            steps_this_batch = self.config["num_game_per_batch"] * self.config["max_eps_length"]
            self.total_env_steps += steps_this_batch
            
            # [New] Custom Logging
            print(f"[{self.config_name}] [{self.start_time}] [Step {self.data['step']}] [Avg Reward: {Avg_reward:.2f}]")

            self.agent.rollout.cal_advantages() 
            
            self.model.train()
            
            for _ in range(self.config["num_epochs"]):
                mini_batch_loader   = self.agent.rollout.mini_batch_generator()
                for mini_batch in mini_batch_loader:

                    for k, v in mini_batch.items():
                        if isinstance(v, torch.Tensor):
                            mini_batch[k] = v.to(self.device)
                            
                    # Move Policy Tuple to Device
                    p_alpha, p_beta = mini_batch["policies"]
                    mini_batch["policies"] = (p_alpha.to(self.device), p_beta.to(self.device))


                    # sliced_memory removed
                    pol_new, val_new = self.model(mini_batch["states"])
                    
                    # Calculate Log Prob and Entropy
                    log_prob_new, entropy = self.dist.log_prob(pol_new, mini_batch["actions"])

                    Kl = self.dist.kl_divergence(mini_batch["policies"], pol_new)
                    
                    
                    # _truly_loss expects flat tensors or compatible broadcasting
                    
                    # Fix shapes
                    val_new = val_new.squeeze(-1) if val_new.dim() > 1 else val_new
                    log_prob_new = log_prob_new # [Batch]
                    
                    # My RolloutBuffer yields flat_advantages which is 1D [Batch]

                    
                    actor_loss, critic_loss, total_loss, entropy = self._standard_ppo_loss(
                        value        = mini_batch["values"].reshape(-1).detach(),
                        value_new    = val_new,
                        entropy      = entropy,
                        log_prob     = mini_batch["probs"].reshape(-1).detach(),
                        log_prob_new = log_prob_new,
                        advantage    = mini_batch["advantages"].reshape(-1).detach(),
                        returns      = mini_batch["returns"].reshape(-1).detach()
                    )
                    
                    # [New] Mono Loss Integration (Diagonal + Directional)
                    loss_lower, loss_upper = self._calculate_mono_loss(mini_batch["states"])
                    
                    # Coefficients
                    coef_lower = self.config.get("mono_coef_lower", 0.1)
                    coef_upper = self.config.get("mono_coef_upper", 0.1)
                    
                    # Total Mono Loss
                    mono_loss = (coef_lower * loss_lower) + (coef_upper * loss_upper)
                    
                    # [New] Critic Warmup Logic
                    # During warmup (first 10,000 steps), freeze Actor and train only Critic.
                    # This stabilizes Value Baseline for SAA Policy (Correction ~ 0).
                    critic_warmup_steps = self.config.get("critic_warmup_steps", 10000)
                    is_warmup = self.data["step"] < critic_warmup_steps
                    
                    if is_warmup:
                        # Zero out Actor-related losses
                        # We still compute them for logging, but remove from backward graph
                        total_loss = self.config["PPO"]["critic_coef"] * critic_loss
                    else:
                        # Full PPO Loss
                        total_loss += mono_loss
                    
                    with torch.autograd.set_detect_anomaly(self.config["set_detect_anomaly"]):
                        if not torch.isnan(total_loss).any():
                            # Dual Step
                            if not is_warmup:
                                self.optimizer_actor.zero_grad()
                            
                            self.optimizer_critic.zero_grad()
                            
                            total_loss.backward()
                            
                            # Clip Grads Separately (Optional, but good practice if separate)
                            # Actor Params
                            if not is_warmup:
                                nn.utils.clip_grad_norm_(self.optimizer_actor.param_groups[0]['params'], max_norm=self.config["max_grad_norm"])
                                self.optimizer_actor.step()
                                
                                # [Mod] Only decay entropy when Actor is training
                                self._entropy_coef_schedule()
                                
                            # Critic Params
                            nn.utils.clip_grad_norm_(self.optimizer_critic.param_groups[0]['params'], max_norm=self.config["max_grad_norm"])
                            self.optimizer_critic.step()
                            
                            # Update Step Count
                            self.data["step"] += 1
                            self.data["entropy_coef"] = self.entropy_coef
                            
                            # Save model periodically (every 200 steps)
                            if self.data["step"] % 200 == 0:
                                self._save_model()

                    if write_data:  
                        with torch.no_grad():
                            try:
                                self.writer.add(
                                        step        = self.data["step"],
                                        Avg_reward    = Avg_reward,
                                        reward      = self.agent.rollout.rewards.mean(), # Accessed direct tensor?
                                        entropy     = entropy,
                                        actor_loss  = actor_loss,
                                        critic_loss = critic_loss,
                                        total_loss  = total_loss,
                                        mono_loss   = mono_loss,   # [New] Log Mono Loss
                                        kl_mean     = Kl.mean().item(),
                                        kl_max      = Kl.max().item(),
                                        kl_min      = Kl.min().item()
                                    )
                            except:
                                pass
             
            self.agent.rollout.reset_data()
            
            # [New] Callback for External Evaluator
            if result_callback is not None:
                 should_stop = result_callback(self, Avg_reward, self.data["step"])
                 if should_stop:
                     print(f"\n[Terminating] External Callback requested stop.")
                     training = False
                     self._save_model()
            
            # Max Update Limit Check
            if self.data["step"] >= self.max_updates:
                print(f"\n[Terminating] Reached maximum training updates ({self.max_updates}). Stopping training.")
                training = False
                self._save_model()


    def _save_model(self):
        """
        Overview:
            Saves the model and other data.
        """
        try:
            torch.save(self.model.cpu().state_dict(), f'{self.save_path}model.pt')
            self.model.to(self.device) # Return to GPU
            with open(f"{self.save_path}stat.json","w") as f:
                json.dump(self.data,f)
            # print(f"Saved model at step {self.data['step']}") # Optional logging
        except Exception as e:
            print(f"[Error] Failed to save model: {e}")


