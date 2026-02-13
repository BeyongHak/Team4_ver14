import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

class Distribution():
    """
    Distribution (Continuous / Gaussian)
    """
    def sample_action(self, policy, action_mask=None):
        """
        Input:
            policy (tuple): (mean, log_std)
        Output:
            action (tensor): Sampled action from Normal(mean, std)
            log_prob (tensor): Log probability
        """
        mean, log_std = policy
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        
        # Use rsample() for reparameterization trick (though PPO usually uses sample, 
        # rsample keeps graph connected if needed, but PPO calculates gradients via log_prob. sample() is fine.
        # But we use sample() here.)
        action = dist.sample()
        
        # Sum log probs over action dimension
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
        
    def log_prob(self, policy, action, action_mask=None):
        mean, log_std = policy
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy

    def kl_divergence(self, policy, policy_new):
        mean, log_std = policy
        dist = Normal(mean, torch.exp(log_std))
        
        mean_new, log_std_new = policy_new
        dist_new = Normal(mean_new, torch.exp(log_std_new))
        
        return kl_divergence(dist, dist_new).sum(dim=-1)

    
