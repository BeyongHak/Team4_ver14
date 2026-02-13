import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, config, state_size, action_size):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Buffer dimensions
        self.num_envs = config["num_game_per_batch"]
        self.max_steps = config["max_eps_length"]
        self.gamma = config["PPO"]["gamma"]
        self.gae_lambda = config["PPO"]["gae_lambda"]

        self.reset_data()

    def reset_data(self):
        # Storage for (Num_Envs, Max_Steps, ...)
        # We fill step by step for all envs.
        self.states = torch.zeros((self.num_envs, self.max_steps, self.state_size), dtype=torch.float32).to(self.device)
        # Continuous Action: (Batch, Action_Dim)
        self.actions = torch.zeros((self.num_envs, self.max_steps, self.action_size), dtype=torch.float32).to(self.device) 
        
        self.log_probs = torch.zeros((self.num_envs, self.max_steps), dtype=torch.float32).to(self.device)
        self.rewards = torch.zeros((self.num_envs, self.max_steps), dtype=torch.float32).to(self.device)
        self.dones = torch.zeros((self.num_envs, self.max_steps), dtype=torch.float32).to(self.device)
        self.values = torch.zeros((self.num_envs, self.max_steps), dtype=torch.float32).to(self.device)
        
        # Store Policy Parameters (Mean, LogStd) for KL calculation
        self.means = torch.zeros((self.num_envs, self.max_steps, self.action_size), dtype=torch.float32).to(self.device)
        self.log_stds = torch.zeros((self.num_envs, self.max_steps, self.action_size), dtype=torch.float32).to(self.device)

        # For Advantage Calculation
        self.advantages = torch.zeros((self.num_envs, self.max_steps), dtype=torch.float32).to(self.device)
        self.returns = torch.zeros((self.num_envs, self.max_steps), dtype=torch.float32).to(self.device)
        
        # Store last values for bootstrapping
        self.last_values = torch.zeros(self.num_envs, dtype=torch.float32).to(self.device)

        self.step_count = 0

    def add_batch_data(self, states, actions, values, rewards, dones, probs, policies):
        """
        Add step data for all environments.
        policies: tuple (mean, log_std)
        """
        if self.step_count >= self.max_steps:
            return
        
        # Unpack policy tuple
        mean, log_std = policies

        self.states[:, self.step_count] = states.to(self.device)
        self.actions[:, self.step_count] = actions.to(self.device) # Shape check might be needed if action is 1D
        self.values[:, self.step_count] = values.to(self.device)
        self.rewards[:, self.step_count] = rewards.to(self.device)
        self.dones[:, self.step_count] = dones.to(self.device)
        self.log_probs[:, self.step_count] = probs.to(self.device)
        
        self.means[:, self.step_count] = mean.to(self.device)
        self.log_stds[:, self.step_count] = log_std.to(self.device)
        
        self.step_count += 1

    def cal_advantages(self):
        """
        Calculate GAE.
        Modified to perform bootstrapping for TimeLimit episodes (Truncated).
        We assume all terminations are due to time limits, so we always bootstrap using the value function.
        """
        last_gae_lam = 0
        for t in reversed(range(self.step_count)):
            if t == self.step_count - 1:
                # Last step: use stored last_values for bootstrapping
                # Ignore done (TimeLimit -> proceed as if it continues)
                next_non_terminal = 1.0 
                next_value = self.last_values
            else:
                # Intermediate steps: use next value from buffer
                # Ignore done (TimeLimit -> proceed as if it continues)
                next_non_terminal = 1.0 
                next_value = self.values[:, t+1]

            delta = self.rewards[:, t] + self.gamma * next_value * next_non_terminal - self.values[:, t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[:, t] = last_gae_lam

        self.returns = self.advantages + self.values

    def mini_batch_generator(self):
        """
        Yields flattened mini-batches.
        """
        # Flatten (Num_Envs, Steps) -> (Num_Envs * Steps)
        batch_size = self.num_envs * self.step_count
        mini_batch_size = batch_size // self.config["n_mini_batches"]

        # Flatten all data
        flat_states = self.states[:, :self.step_count].reshape(-1, self.state_size)
        flat_actions = self.actions[:, :self.step_count].reshape(-1, self.action_size) # Continuous Action is Vector
        flat_probs = self.log_probs[:, :self.step_count].reshape(-1)
        flat_advantages = self.advantages[:, :self.step_count].reshape(-1)
        flat_values = self.values[:, :self.step_count].reshape(-1)
        flat_returns = self.returns[:, :self.step_count].reshape(-1)
        
        # Flatten Policy Args
        flat_means = self.means[:, :self.step_count].reshape(-1, self.action_size)
        flat_log_stds = self.log_stds[:, :self.step_count].reshape(-1, self.action_size)

        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_indices = indices[start:end]

            yield {
                "states": flat_states[mb_indices],
                "actions": flat_actions[mb_indices],
                "probs": flat_probs[mb_indices],
                "advantages": flat_advantages[mb_indices],
                "values": flat_values[mb_indices],
                "returns": flat_returns[mb_indices],
                "policies": (flat_means[mb_indices], flat_log_stds[mb_indices]) # Re-pack as Tuple
            }
