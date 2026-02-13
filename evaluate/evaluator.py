
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from collections import deque

class Evaluator:
    def __init__(self, config, env_creator, save_path, device, 
                 eval_interval=5000, patience=10, min_steps=100000, scenario_path=None):
        """
        Args:
            config: Eval environment config
            env_creator: Function to create eval environment (should return SubprocVecEnv)
            save_path: Path to save best_model.pt
            device: torch device
            eval_interval: Training steps between evaluations (default 5000)
            patience: Early stopping patience (default 10)
            min_steps: Minimum training steps before enabling early stopping (default 150000)
            scenario_path: Path to evaluation scenarios (.npy files)
        """
        self.config = config
        self.env_creator = env_creator
        self.save_path = save_path
        self.device = device
        
        self.eval_interval = eval_interval
        self.patience = patience
        self.min_steps = min_steps
        
        # State Tracking
        self.best_global_score = float('inf') # Lower cost is better
        self.no_improvement_count = 0
        
        self.interval_best_train_reward = -float('inf')
        self.interval_best_state = None
        
        # Load Scenarios
        self.scenarios = self._load_scenarios(scenario_path)
        
        # Create Eval Env (Lazy init or immediate)
        self.eval_env = None
        if self.scenarios:
             self.eval_env = self.env_creator()

    def _load_scenarios(self, path):
        scenarios = []
        if path and os.path.exists(path):
            files = sorted(glob.glob(os.path.join(path, "*.npy")))
            # Take first 10 scenarios
            files = files[:10]
            for f in files:
                try:
                    data = np.load(f, allow_pickle=True)
                    scenarios.append(data)
                except:
                    pass
            print(f"[Evaluator] Loaded {len(scenarios)} scenarios from {path}")
        else:
            print(f"[Evaluator] Warning: Scenario path {path} not found.")
        return scenarios

    def on_train_step(self, trainer, avg_reward, step):
        """
        Callback triggered during training loop.
        Returns True if training should stop.
        """
        
        # 1. Update Candidate (Best Avg Reward within this interval)
        if avg_reward > self.interval_best_train_reward:
            self.interval_best_train_reward = avg_reward
            # Deep copy model state
            self.interval_best_state = {k: v.clone().cpu() for k, v in trainer.model.state_dict().items()}

        # 2. Check Interval
        if step >= self.min_steps and step % self.eval_interval == 0:
            should_stop = self._run_evaluation(trainer, step)
            
            # Reset Interval Tracker
            self.interval_best_train_reward = -float('inf')
            self.interval_best_state = None
            
            return should_stop
            
        return False

    def _run_evaluation(self, trainer, step):
        print(f"\n[Evaluator] Step {step}: Evaluating Interval Candidate (Train Reward: {self.interval_best_train_reward:.2f})...")
        
        if self.interval_best_state is None:
            print("[Evaluator] No candidate found (unexpected). Skipping.")
            return False

        if not self.eval_env or not self.scenarios:
            print("[Evaluator] No env or scenarios. Skipping.")
            return False

        # 1. Backup Current Model
        current_state_backup = {k: v.clone().cpu() for k, v in trainer.model.state_dict().items()}
        
        # 2. Load Candidate State
        trainer.model.load_state_dict(self.interval_best_state)
        trainer.model.to(self.device)
        
        # 3. Evaluate on Fixed Scenarios
        try:
             eval_cost = self._evaluate_on_scenarios(trainer.model)
        except Exception as e:
             print(f"[Evaluator] Error during evaluation: {e}")
             eval_cost = float('inf')

        print(f"[Evaluator] Candidate Avg Cost: {eval_cost:.4f} (Global Best: {self.best_global_score:.4f})")
        
        # 4. Compare with Global Best
        if eval_cost < self.best_global_score:
            print(f" >>> WINNER! Updating Best Model (Improved by {self.best_global_score - eval_cost:.4f})")
            print(f" >>> [Comparison] Old Best: {self.best_global_score:.4f} vs New Candidate: {eval_cost:.4f}")
            self.best_global_score = eval_cost
            self.no_improvement_count = 0
            print(f" >>> Saving to {self.save_path}best_model.pt ...")
            torch.save(trainer.model.cpu().state_dict(), f'{self.save_path}best_model.pt')
            trainer.model.to(self.device) # Restore device
        else:
            diff = eval_cost - self.best_global_score
            print(f" >>> RETAIN! Candidate failed to beat Best Model (Worse by {diff:.4f})")
            print(f" >>> [Comparison] Current Best: {self.best_global_score:.4f} vs Candidate: {eval_cost:.4f}")
            
            if step >= self.min_steps:
                self.no_improvement_count += 1
                print(f" >>> [Patience] No Improvement Count: {self.no_improvement_count} / {self.patience}")
            else:
                 print(f" >>> [Patience] Reset/Ignored. (Warmup Phase: Step {step} < {self.min_steps})")

        # 5. Restore Training Model
        trainer.model.load_state_dict(current_state_backup)
        trainer.model.to(self.device)
        
        # 6. Early Stopping Check
        if step >= self.min_steps and self.no_improvement_count >= self.patience:
            print(f"\n[Evaluator] Early Stopping Triggered at Step {step}!")
            return True # Stop Training
            
        return False

    def _evaluate_on_scenarios(self, model):
        """
        Internal evaluation loop on fixed scenarios
        """
        num_scenarios = len(self.scenarios)
        
        # Distribute scenarios
        if hasattr(self.eval_env, 'num_envs'):
             scenarios_to_run = self.scenarios[:self.eval_env.num_envs]
        else:
             scenarios_to_run = self.scenarios

        # Set Fixed Demands
        args_list = [(s,) for s in scenarios_to_run]
        self.eval_env.call_method_batch("set_fixed_demands", args_list)
        
        obs = self.eval_env.reset()
        
        # [Robustness] Determine max steps from scenario length
        if len(scenarios_to_run) > 0:
            # scenarios are usually [time, retailers]
            scenario_len = len(scenarios_to_run[0])
            max_len = scenario_len
            print(f"[Evaluator] Using dynamic Max Steps from scenario: {max_len}")
            
            # [Fix] Sync Environment Max Steps with Scenario Length
            # Call set_max_steps on all environments
            self.eval_env.call_method_batch("set_max_steps", [(max_len,)] * len(scenarios_to_run))
        else:
            max_len = self.config.get("max_eps_length", 1000)
            print(f"[Evaluator] Warning: No scenario found. Using config Max Steps: {max_len}")
        
        total_costs = np.zeros(len(scenarios_to_run))
        active_steps = np.zeros(len(scenarios_to_run))
        dones_recorded = np.zeros(len(scenarios_to_run), dtype=bool)
        
        model.eval()
        
        for step_idx in range(max_len):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                ((mean, log_std), _) = model(obs_tensor)
                
                # [Standard Forward Pass]
                # The model's 'mean' is the 'Correction' (Delta) value.
                # True Action = SAA_Base - Correction
                
                # 1. Get SAA Base
                saa_base = model.get_saa_base_action(obs_tensor)
                
                # 2. Apply Correction (Using Mean for Deterministic Eval)
                raw_target = saa_base - mean
                
                # 3. Clamp
                final_action = torch.clamp(raw_target, min=0.0)
                
                action = final_action.cpu().numpy()
            
            obs, rewards, dones, infos = self.eval_env.step(action)
            
            active_mask = ~dones_recorded
            step_costs = -rewards * self.config.get("reward_scale", 100.0)
            
            total_costs[active_mask] += step_costs[active_mask]
            active_steps[active_mask] += 1
            
            new_dones = dones.astype(bool)
            dones_recorded = dones_recorded | new_dones
            
            if np.all(dones_recorded):
                print(f"[Evaluator_Debug] All scenarios done at step {step_idx + 1}")
                break
                
        model.train()
        
        # Calculate Average Cost per Episode
        # [User Request] Return Step Average Cost because Eval/Train lengths differ
        # Use actual steps run for EACH scenario to handle variable lengths or early cutoffs
        
        # Avoid division by zero
        active_steps[active_steps == 0] = 1.0
        avg_cost_per_step = total_costs / active_steps 
        
        final_score = np.mean(avg_cost_per_step)
        
        # Cleanup
        self.eval_env.call_method_batch("set_fixed_demands", [(None,)] * len(scenarios_to_run))
        
        return final_score
