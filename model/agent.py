import torch
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical

from model.rollout_buffer import RolloutBuffer
from model.distribution import Distribution

class Agent():
    """Agent"""
    def __init__(self,env,model,config,device):
        super().__init__()
        self.env           = env
        self.model         = model
        self.device        = device         
        # self.reward        = config["rewards"] # Removed unused key that caused error
        self.rollout       = RolloutBuffer(config,env.getStateSize(),env.getActionSize())
        self.dist          = Distribution()

    @torch.no_grad()
    def play(self, state):
        """
        Vectorized play.
        state: (Batch_Size, State_Dim)
        """
        # [Modified] Removed Try-Except Block for strict debugging
        tensor_state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        # MLP Model Forward: Returns ((mean, log_std), value)
        policy, value = self.model(tensor_state) 
        
        # Sample Correction Amount (Gaussian)
        # We interpret this sample as the "Adjustment Amount"
        correction, log_prob = self.dist.sample_action(policy)
        
        # [New] Get SAA Base Action
        saa_base_action = self.model.get_saa_base_action(tensor_state)
        
        # [Residual Logic]
        # User removed F.relu(correction) to allow correction to be negative.
        # If correction is negative, Action > SAA is possible.
        raw_target = saa_base_action - correction
        
        final_action = torch.clamp(raw_target, min=0.0)
        
        # Handle Tuple Policy for CPU storage (store params)
        p_mean, p_logstd = policy
        policy_cpu = (p_mean.detach().cpu(), p_logstd.detach().cpu())
        
        step_data = {
            "state": torch.from_numpy(state),
            "value": value.detach().cpu(), 
            "prob": log_prob.detach().cpu(),
            "policy": policy_cpu,
            "action_sample": correction.detach().cpu() # Store the Gaussian sample (correction)
        }
        
        return final_action.cpu().numpy(), step_data

    def run(self, num_games):
        """
        Vectorized run.
        """
        # We assume self.env is a VecEnv
        obs = self.env.reset() # (Batch, State)
        
        # Reset rollout counters
        self.rollout.game_count = 0 
        self.rollout.step_count = 0
        self.rollout.reset_data() # Ensure buffer is clear
        
        total_rewards = np.zeros(self.env.num_envs)
        
        # Run for the full length of an episode
        for step in range(self.rollout.max_steps):
            action, step_data = self.play(obs)
            
            # Step the environment (Action is Scaled S_RL)
            next_obs, rewards, dones, infos = self.env.step(action)
            
            if step_data is not None:
                # [Modified] Store 'action_sample' (correction) in rollout buffer
                # because PPO optimizes the Gaussian distribution producing this sample.
                stored_actions = step_data["action_sample"]
                
                self.rollout.add_batch_data(
                    states        = step_data["state"],
                    actions       = stored_actions, # Storing Correction
                    values        = step_data["value"].squeeze(),
                    rewards       = torch.from_numpy(rewards).float(),
                    dones         = torch.from_numpy(dones).float(),
                    probs         = step_data["prob"],
                    policies      = step_data["policy"]
                )
            
            total_rewards += rewards
            obs = next_obs
            
            if np.all(dones):
                break
        
        # Calculate Last Value for Bootstrapping (Time Limit)
        with torch.no_grad():
            tensor_obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            _, last_values = self.model(tensor_obs)
            self.rollout.last_values = last_values.squeeze().detach()

        avg_reward = np.mean(total_rewards)
        # print(f"Batch Finished. Avg Reward: {avg_reward:.4f}")
        return avg_reward

