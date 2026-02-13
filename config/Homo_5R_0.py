config = {
    'mono_coef_lower': 0.1,
    'mono_coef_upper': 0.1,

    "max_eps_length": 1000,
    "lead_time": 0,
    "holding_cost": 1.0,
    "backorder_cost": 4.0,
    "fixed_cost": 0.0,
    "reward_scale": 100.0,

    "retailers": [
        {'delta': 60, 'mean': 10, 'std': (10)**(0.5)},
        {'delta': 60, 'mean': 10, 'std': (10)**(0.5)},
        {'delta': 60, 'mean': 10, 'std': (10)**(0.5)},
        {'delta': 60, 'mean': 10, 'std': (10)**(0.5)},
        {'delta': 60, 'mean': 10, 'std': (10)**(0.5)},
    ],

    "network_arch": {

        "embedding_dim_A": 32,
        "embedding_dim_C": 32,

        "mlp_inv_layers": [128, 64, 32],
        "mlp_ret_layers": [64, 32, 32],
        "policy_layers": [128, 128, 64, 32],
        "value_layers": [128, 128, 64],
    },

    "PPO": {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "policy_kl_range": 0.0008,
        "policy_params": 20,
        "value_clip": 0.2,
        "critic_coef": 0.5,
    },

    "entropy_coef": {
        "start":  0.005,
        "end": 0.0001,
        "step": 100_000
    },

    "num_epochs": 10,
    "lr": 1e-4,
    "num_game_per_batch": 16,
    "max_grad_norm": 0.5,
    "n_mini_batches": 10,

    "set_detect_anomaly": False,
    "normalize_advantage": True
}
