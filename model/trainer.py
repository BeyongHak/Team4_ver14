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
        Calculate Diagonal Monotonicity Loss for ALL states along the path.
        Vectorized Implementation (No Loop).
        
        Path:
        - Lower: All integer steps d down from S_curr until base state (at least one 0).
        - Upper: All integer steps d up from S_curr until final state (at least one Max K).
        
        Constraints:
        - Lower Path: Action(S_lower_d) <= Action(S_curr). Loss if >. Gradient on S_curr.
        - Upper Path: Action(S_curr) <= Action(S_upper_d). Loss if >. Gradient on S_upper_d.
        """
        batch_size = states.shape[0]
        device = states.device
        
        # 1. Setup & Dimensions
        inv_input = states[:, :self.model.inv_dim] # (B, Inv)
        ret_input = states[:, self.model.inv_dim:] # (B, N)
        
        # Max K Tensor
        max_k_tensor = self.model.retailer_max_k_tensor.to(device) # (N,)
        
        # Current Integer State
        k_int = torch.round(ret_input * max_k_tensor) # (B, N)
        
        # 2. Determine Maximum Steps needed for the Batch
        # We need the max possible steps any state in the batch can take.
        
        # Lower: Max steps down = min(k_int) across retailers
        max_steps_down_per_sample = k_int.min(dim=1).values # (B,)
        global_max_down = int(max_steps_down_per_sample.max().item())
        
        # Upper: Max steps up = min(Max K - k_int) across retailers
        dist_to_max = max_k_tensor - k_int
        max_steps_up_per_sample = dist_to_max.min(dim=1).values # (B,)
        global_max_up = int(max_steps_up_per_sample.max().item())
        
        # 3. Create Step Grid (Broadcasting)
        # We create a range [1, 2, ..., Max_Step]
        
        # --- Lower Path Expansion ---
        loss_lower = torch.tensor(0.0, device=device)
        
        if global_max_down > 0:
            # Range: (1, Global_Max_Down)
            steps_range = torch.arange(1, global_max_down + 1, device=device).float() # (M,)
            # Expand to (B, M)
            steps_grid = steps_range.unsqueeze(0).expand(batch_size, -1) # (B, M)
            
            # Mask: Is step d valid for sample b? (d <= max_steps_down_b)
            mask_lower = (steps_grid <= max_steps_down_per_sample.unsqueeze(1)) # (B, M)
            
            if mask_lower.any():
                # Prepare Inputs
                # S_lower = S_curr - d
                # Reshape Ret: (B, 1, N)
                # Reshape Steps: (B, M, 1)
                # Max K: (1, 1, N)
                
                ret_expanded = ret_input.unsqueeze(1) # (B, 1, N)
                inv_expanded = inv_input.unsqueeze(1).expand(-1, global_max_down, -1) # (B, M, Inv) (Duplicate Inv)
                
                delta = steps_grid.unsqueeze(2) / max_k_tensor.view(1, 1, -1) # (B, M, N)
                
                ret_lower_grid = ret_expanded - delta
                ret_lower_grid = torch.clamp(ret_lower_grid, min=0.0)
                
                # Merge to State: (B, M, Inv+Ret) -> Flatten to (B*M, State)
                # We process all B*M states at once
                
                flat_inv = inv_expanded.reshape(-1, self.model.inv_dim)
                flat_ret = ret_lower_grid.reshape(-1, self.model.ret_dim)
                flat_states_lower = torch.cat([flat_inv, flat_ret], dim=1)
                
                # Get Actions
                # Action(Lower) - No Grad
                with torch.no_grad():
                    # Helper logic inline to avoid overhead
                    saa_lower = self.model.get_saa_base_action(flat_states_lower)
                    (mean_lower, _), _ = self.model(flat_states_lower)
                    act_lower = F.relu(saa_lower - mean_lower)
                    act_lower = act_lower.view(batch_size, global_max_down, -1) # (B, M, 1)
                    
                # Action(Curr) - With Grad (But repeated M times for comparison)
                # We need Grad flow to S_curr
                # Re-compute S_curr Action to attach graph or use expanded
                # Better to compute once and expand
                saa_curr = self.model.get_saa_base_action(states)
                (mean_curr, _), _ = self.model(states)
                act_curr = F.relu(saa_curr - mean_curr) # (B, 1)
                
                act_curr_expanded = act_curr.unsqueeze(1).expand(-1, global_max_down, -1) # (B, M, 1)
                
                # Loss Calculation
                # Constraint: Action(Lower) <= Action(Curr)
                # Violation: Action(Lower) > Action(Curr)
                # Grad: Increase Action(Curr)
                
                diff_lower = act_lower - act_curr_expanded
                loss_grid = F.relu(diff_lower) # (B, M, 1) [Mod] L1 Penalty (Linear)
                
                # Apply Mask
                loss_grid = loss_grid * mask_lower.unsqueeze(2).float()
                
                # Mean over valid entries
                # [Mod] Batch Mean: Sum over steps / Batch Size
                # We want to punish MORE if there are MORE violations along the path.
                # So we do NOT divide by valid_count (steps), but by batch_size.
                
                loss_lower = loss_grid.sum() / batch_size
        
        # --- Upper Path Expansion ---
        loss_upper = torch.tensor(0.0, device=device)
        
        if global_max_up > 0:
            steps_range = torch.arange(1, global_max_up + 1, device=device).float()
            steps_grid = steps_range.unsqueeze(0).expand(batch_size, -1)
            
            mask_upper = (steps_grid <= max_steps_up_per_sample.unsqueeze(1))
            
            if mask_upper.any():
                ret_expanded = ret_input.unsqueeze(1) # (B, 1, N)
                inv_expanded = inv_input.unsqueeze(1).expand(-1, global_max_up, -1)
                
                delta = steps_grid.unsqueeze(2) / max_k_tensor.view(1, 1, -1)
                
                ret_upper_grid = ret_expanded + delta
                # Soft upper limit; model should handle >1.0 inputs if scaling allows, 
                # but let's assume valid range is handled by NN.
                
                flat_inv = inv_expanded.reshape(-1, self.model.inv_dim)
                flat_ret = ret_upper_grid.reshape(-1, self.model.ret_dim)
                flat_states_upper = torch.cat([flat_inv, flat_ret], dim=1)
                
                # Get Actions
                # Action(Upper) - With Grad (We want to change Upper)
                # Action(Curr) - No Grad (Anchor)
                
                # Re-compute Upper Actions (Batch * M)
                saa_upper = self.model.get_saa_base_action(flat_states_upper)
                (mean_upper, _), _ = self.model(flat_states_upper)
                act_upper = F.relu(saa_upper - mean_upper)
                act_upper = act_upper.view(batch_size, global_max_up, -1)
                
                # Current Action (No Grad)
                with torch.no_grad():
                    saa_curr = self.model.get_saa_base_action(states)
                    (mean_curr, _), _ = self.model(states)
                    act_curr = F.relu(saa_curr - mean_curr) 
                    
                act_curr_expanded = act_curr.unsqueeze(1).expand(-1, global_max_up, -1)
                
                # Loss Calculation
                # Constraint: Action(Curr) <= Action(Upper)
                # Violation: Action(Curr) > Action(Upper)
                # Grad: Increase Action(Upper)
                
                diff_upper = act_curr_expanded - act_upper
                loss_grid = F.relu(diff_upper) # [Mod] L1 Penalty (Linear)
                
                loss_grid = loss_grid * mask_upper.unsqueeze(2).float()
                
                loss_upper = loss_grid.sum() / batch_size

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


