
import os
import argparse
import torch
import numpy as np
import random
import importlib
import psutil
import multiprocessing as mp
from scipy.stats import poisson

# Environments
from env.env_OWMR_continuous import ManufacturerOWMREnv
from env.env_OWMR_discrete_action import ManufacturerOWMRDiscreteActionEnv
# from env.retailer import Retailer <--- Removed
from env.vector_env import SubprocVecEnv, DummyVecEnv
from env.truncNormal import HybridTruncNorm

# Trainer
from model.trainer import Trainer
from evaluate.evaluator import Evaluator

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_env(config, seed, is_discrete):
    """
    Factory function to create an environment instance.
    Dynamically selects Discrete vs Continuous based on flag.
    """
    def _thunk():
        # Set seed for this specific process/env
        np.random.seed(seed)
        random.seed(seed)
        
        # Re-create retailers here to ensure they are fresh in the process
        retailer_objects = []
        for r_conf in config['retailers']:
            delta = r_conf['delta']
            mean_demand = r_conf['mean']
            
            if is_discrete:
                # [Discrete Mode] Use Poisson
                customer_dist = poisson(mu=mean_demand)
            else:
                # [Continuous Mode] Use HybridTruncNorm
                # Get std or default to 20% of mean
                std_demand = r_conf.get('std', mean_demand * 0.2) 
                customer_dist = HybridTruncNorm(mean=mean_demand, std=std_demand, lower=0)
            
            # [Refactor] Use dict instead of Retailer object
            r = {'delta': delta, 'customer': customer_dist}
            retailer_objects.append(r)
            
        if is_discrete:
            # [Discrete Mode]
            env = ManufacturerOWMRDiscreteActionEnv(config, retailers=retailer_objects)
        else:
            # [Continuous Mode]
            env = ManufacturerOWMREnv(config, retailers=retailer_objects)
            
        return env
    return _thunk

def main(args):
    # 1. Config Load
    try:
        config_module = importlib.import_module(f"config.{args.config}")
    except ImportError:
        print(f"[Error] Could not load config '{args.config}'. Check if it exists in 'config/'.")
        return

    config = config_module.config.copy() 
    config["config_name"] = args.config 
    
    # 2. Determine Mode (Discrete vs Continuous) based on Config Name prefix
    is_discrete = args.config.startswith("D_")
    
    # 3. Experiment Setup & Paths
    base_path = "./"
    
    if is_discrete:
        mode_str = "Discrete Action/Demand"
        save_dir = f"saved_model_D"
        run_dir = f"run_D"
    else:
        mode_str = "Continuous Action/Demand"
        save_dir = f"saved_model_C"
        run_dir = f"run_C"
        
    save_path = os.path.join(base_path, save_dir, args.exp_name) + "/"
    writer_path = os.path.join(base_path, run_dir, args.exp_name)
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(writer_path, exist_ok=True)

    print(f"==================================================")
    print(f" Start Training: {args.exp_name}")
    print(f" Config: {args.config}")
    print(f" Mode: {mode_str}")
    print(f" Num Envs: {args.num_envs}")
    print(f" Max Updates: {args.max_updates}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Device: {device}")
    print(f"==================================================")

    if args.seed is not None:
        set_seed(args.seed)

    # 4. Create Vector Environment
    # Update config to match num_envs
    # Trainer uses "num_game_per_batch" to dimension the buffer.
    config["num_game_per_batch"] = args.num_envs
    config["eval_steps"] = args.eval_steps  # [New] Pass eval steps to config
    
    # Create env functions
    # Pass 'is_discrete' flag to make_env
    env_fns = [make_env(config, args.seed + i, is_discrete) for i in range(args.num_envs)]
    
    # Start Parallel Environments
    # Start Parallel Environments
    # [Fix] Switch to DummyVecEnv (Sequential) for Training to prevent Cloud Deadlocks (SubprocVecEnv)
    # The user reported hangs during initialization. Sequential is safer.
    print(f" initializing {args.num_envs} environments (Sequential/Dummy)...")
    vec_env = DummyVecEnv(env_fns)
    # vec_env = SubprocVecEnv(env_fns) # Original
    print(" Vector Environment initialized successfully.")

    # 5. Initialize Trainer
    trainer = Trainer(
        config=config,
        env=vec_env,
        writer_path=writer_path,
        save_path=save_path,
        max_updates=args.max_updates
    )

    # 6. Initialize Evaluator (External)
    scenario_path = os.path.join("evaluate", args.scenario) if args.scenario else None
    
    # Define an env creator for the evaluator (Lazy initialization)
    def eval_env_creator():
        print(" initializing parallel evaluation environments (10 envs)...")
        num_eval_envs = 10
        eval_config = config.copy()
        # [Fix] Force max_eps_length to match eval_steps for Evaluation Envs
        eval_config["max_eps_length"] = args.eval_steps
        
        eval_env_fns = [make_env(eval_config, seed=10000+i, is_discrete=is_discrete) for i in range(num_eval_envs)]
        # [Fix] Use DummyVecEnv (Sequential) for Evaluation to avoid MP overhead/deadlocks
        print(f" [Evaluator] Using DummyVecEnv (Sequential) for {num_eval_envs} envs.")
        return DummyVecEnv(eval_env_fns)

    entropy_end_step = config.get("entropy_coef", {}).get("step", 100000)

    evaluator = Evaluator(
        config=config,
        env_creator=eval_env_creator,
        save_path=save_path,
        device=device,
        eval_interval=args.eval_interval,
        patience=args.patience,
        min_steps=entropy_end_step, # Sync with Entropy Decay Step
        scenario_path=scenario_path
    )

    # 7. Start Training
    try:
        trainer.train(result_callback=evaluator.on_train_step)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        trainer._save_model()
        print("Model saved.")
    finally:
        vec_env.close()

if __name__ == "__main__":
    # Windows Support for Multiprocessing
    mp.freeze_support()
    
    parser = argparse.ArgumentParser(description="Unified PPO Training (Continuous/Discrete)")
    parser.add_argument("--config", type=str, required=True,
                        help="Config file name (e.g. Homo_2R_6 or D_Homo_2R_6)")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name (defaults to config name)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--scenario", type=str, default=None, 
                        help="Name of the scenario set in 'evaluate/' folder")
    
    # Evaluation & Early Stopping Args
    parser.add_argument("--eval-interval", type=int, default=5000, 
                        help="Evaluation interval in update steps (default: 5000)")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--max-updates", type=int, default=1000000, 
                        help="Maximum training updates (default: 1M)")
    parser.add_argument("--eval-steps", type=int, default=10000,
                        help="Evaluation scenario duration (steps). Default: 10000.")

    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = args.config

    main(args)
