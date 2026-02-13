
import multiprocessing as mp
import os
import sys
import argparse
import glob
import numpy as np
import torch
import importlib
import csv
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.getcwd())
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from model.model import PPOMLPModel
from env.env_OWMR_continuous import ManufacturerOWMREnv as EnvOWMRContinuous
from env.env_OWMR_discrete_action import ManufacturerOWMRDiscreteActionEnv as EnvOWMRDiscreteAction
from env.truncNormal import HybridTruncNorm
from scipy.stats import poisson

def get_experiment_list(mode="all", filters=None, selects=None):
    """
    Reusing logic from run_batch_experiments.py to discover configs.
    """
    pattern = "*.py" 
    config_dir = "config"
    files = glob.glob(os.path.join(config_dir, pattern))
    experiments = []

    for f in files:
        filename = os.path.basename(f)
        if filename == "__init__.py": continue
        
        config_name = filename.replace(".py", "")
        if selects or filters:
            match = False
            
            # 1. Exact Match (Select)
            if selects:
                if config_name in selects:
                    match = True
            
            # 2. Substring Match (Filter)
            if not match and filters:
                for term in filters:
                    term_clean = term.replace(".py", "")
                    if term_clean in config_name:
                        match = True
                        break
            
            if not match: continue
        
        is_discrete = config_name.startswith("D_")
        if mode == "discrete" and not is_discrete: continue
        if mode == "continuous" and is_discrete: continue
        
        # Determine Scenario Name
        # Logic: Strip known output suffixes (like _ReLU) first, then numeric suffixes
        base_name = config_name
        if base_name.lower().endswith("_relu"):
            base_name = base_name[:-5] # remove _ReLU
            
        parts = base_name.split('_')
        while parts and parts[-1].isdigit():
            parts.pop()
        scenario_name = "_".join(parts)

        experiments.append({
            "config": config_name,
            "scenario": scenario_name,
            "is_discrete": is_discrete
        })
    
    # Sort
    experiments.sort(key=lambda x: x['config'])
    return experiments

def load_config(config_name):
    module = importlib.import_module(f"config.{config_name}")
    return module.config

def parse_config_info(config_name):
    """
    Extracts LeadTime, NumRetailers, Heterogeneity from config name.
    """
    parts = config_name.replace("D_", "").split('_')
    
    is_hetero = "Hetero"
    if "Homo" in parts: is_hetero = "Homo"
    
    num_retailers = 0
    # Heuristic: 5R -> 5
    for p in parts:
        if "R" in p and p[:-1].isdigit():
            num_retailers = int(p[:-1])
            
    return is_hetero, num_retailers

def get_saa_group_id(config_name):
    """
    Extracts the SAA Group ID (Problem Setting) from the config name.
    e.g. Homo_5R_8_0_0 -> Homo_5R_8
    """
    seg = config_name.split('_')
    r_idx = -1
    for i, p in enumerate(seg):
        if p.endswith('R') and p[:-1].isdigit():
            r_idx = i
            break
            
    # Typically Format: [D], Type, NumR, LeadTime, ...
    # We want up to LeadTime.
    # r_idx points to NumR.
    # LeadTime is usually r_idx + 1.
    
    if r_idx != -1 and r_idx + 1 < len(seg):
        return "_".join(seg[:r_idx+2])
        
    return config_name

def calculate_ci(mean, std, n):
    if n <= 1: return mean, mean
    t_crit = stats.t.ppf(0.975, df=n-1)
    margin = t_crit * (std / np.sqrt(n))
    return mean - margin, mean + margin


import itertools

def generate_full_policy_map(env, model, device, config_name, save_path, is_discrete):
    """
    Scans the Policy to generate a lookup table.
    Per user request, this now logs EVERY state combination individually (no aggregation for Homo).
    It saves the raw Action (Target Level).
    """
    
    print(f"  [Policy Scan] Generating Map for {config_name} (Raw States)...")

    # 2. Define State Coverage via SAA Max K (Available in Env)
    # Env has already loaded the SAA map and determined max K per retailer.
    # user request: Limit = Max K + 1 (so we iterate up to Max K + 1)
    # range(N) goes 0..N-1. So to include (Max K + 1), we need range(Max K + 2).
    
    limits = [int(k) + 2 for k in env.retailer_max_k]
    
    total_combs = 1
    for l in limits: total_combs *= l
    
    print(f"  [Scan Limits] Retailer Max Ks: {[int(k) for k in env.retailer_max_k]}")
    print(f"  [Scan Limits] Loop Ranges: {limits} -> Total Combs: {total_combs}")
    
    # Check feasibility
    if total_combs > 1000000: # Increased limit slightly as valid states are sparse? 
        # Actually itertools.product is eager or lazy? Product is lazy.
        # But we listify it below. 1M is manageable for 1024 batch.
        print(f"  [Warning] Large state space ({total_combs}). This might take a while.")
        
    if total_combs > 5000000:
         print(f"  [Skip] Too many state combinations ({total_combs}). Skipping full map generation.")
         return

    # Generate all states
    ranges = [range(k) for k in limits]
    all_states = itertools.product(*ranges)
    
    # Prepare Model Input
    inv_part_len = 2 + env.lead_time
    base_inv = np.zeros(inv_part_len, dtype=np.float32)
    
    # Normalizer: Use Env's detected Max K (derived from SAA)
    retailer_max_k_norm = np.array(env.retailer_max_k, dtype=np.float32)
    max_inv = getattr(env, 'max_inventory', 500)
    # min_a removed

    # Storage
    # Key: Tuple (Raw State), Value: Action
    policy_output = {}

    # Batch Processing
    batch_size = 1024
    all_states_list = list(all_states)
    
    # Pre-calculate normalizer
    retailer_max_k_norm_t = torch.tensor(retailer_max_k_norm, device=device).unsqueeze(0) # (1, num_ret)
    base_inv_t = torch.tensor(base_inv, device=device).unsqueeze(0) # (1, inv_len)

    for i in tqdm(range(0, len(all_states_list), batch_size), desc="Scanning (Batched)"):
        batch_states = all_states_list[i : i + batch_size]
        
        # Prepare Batch Tensor
        raw_k_batch = np.array(batch_states, dtype=np.float32) # (B, num_ret)
        raw_k_t = torch.tensor(raw_k_batch, device=device)
        
        # Normalize: raw / max
        norm_k_t = raw_k_t / retailer_max_k_norm_t # (B, num_ret)
        
        # Concatenate: [base_inv, norm_k]
        # Expand base_inv to batch
        base_inv_batch = base_inv_t.expand(len(batch_states), -1)
        obs_tensor = torch.cat([base_inv_batch, norm_k_t], dim=1)
        
        with torch.no_grad():
            # Output: ((mean, log_std), value)
            (mean, log_std), _ = model(obs_tensor)
            saa_base = model.get_saa_base_action(obs_tensor)
            
        # To CPU
        mean_np = mean.cpu().numpy().flatten()
        log_std_np = log_std.cpu().numpy().flatten()
        saa_np = saa_base.cpu().numpy().flatten()
        
        # Decode & Store
        for j, state_tuple in enumerate(batch_states):
            correction_val = float(mean_np[j])
            std_val = float(np.exp(log_std_np[j]))
            saa_val = float(saa_np[j])
            
            # Logic: Target = SAA - Correction
            raw_target = saa_val - correction_val
            final_action = max(0.0, raw_target)
            # final_action = max(0.0, min(saa_val, raw_target)) <-- REMOVED min
            
            if is_discrete:
                final_action = int(np.round(final_action))
            
            # Storing: Action, Correction, SAA, LogStd
            policy_output[state_tuple] = (final_action, correction_val, saa_val, log_std_np[j])

    # 3. Write Compiled CSV (DRL_policy_map.csv)
    # Format: State_Key, Action_Target_Level
    csv_path = os.path.join(save_path, f"DRL_policy_map.csv")
    
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        # Modified columns for Gaussian as user requested
        # Key, SAA, Residual(Correction), Real_Action, Log_Std
        w.writerow(["Retailer_State_Pair", "SAA_Policy_Val", "Residual_Correction", "Real_Action_Target", "Log_Std"])
        
        sorted_keys = sorted(policy_output.keys())
        for k in sorted_keys:
            # policy_output stored as: (final_action, correction_val, saa_val, log_std_val)
            # wait, previous storage was: (final_action, correction_val, correction_val, log_std) - duplicated correction
            # checking previous storage...
            # Step Id 182: (final_action, correction_val, correction_val, log_std_np[j]) - Yes duplicate
            # I will modify the storage line too in another chunk.
            
            # Assuming storage is updated: 
            action_val, corr_val, saa_val, log_std_val = policy_output[k]
            clean_key = [int(x) for x in k]
            w.writerow([clean_key, f"{saa_val:.2f}", f"{corr_val:.6f}", f"{action_val:.2f}", f"{log_std_val:.4f}"])
            
    print(f"  [Map] Saved DRL Policy Map to {csv_path}")
import itertools

# SAA Policy Class
class SAAPolicy:
    def __init__(self, config_name, num_retailers, is_discrete, save_path=None):
        self.num_retailers = num_retailers
        self.is_discrete = is_discrete
        self.config_name = config_name
        self.is_hetero = "Hetero" in config_name or "D_Hetero" in config_name
        self.is_homo = not self.is_hetero
        self.save_path = save_path
        
        # Determine CSV Path from baseline
        
        base_dir = os.getcwd() # numerical.py runs from root
        base_dir = os.getcwd() # numerical.py runs from root
        
        # [User Request] Strict Mode: Only check policy_SAA/{config_name}/SAA_policy_map.csv
        # Fallback to baseline is REMOVED to prevent unintended behavior.
        
        csv_path = os.path.join(base_dir, "policy_SAA", config_name, "SAA_policy_map.csv")
        
        if os.path.exists(csv_path):
            print(f"  [SAA] Loading Policy Map: {csv_path}")
        else:
             # Strict Error
             raise FileNotFoundError(f"SAA Policy Map NOT found at strict path: {csv_path}\n"
                                     f"Please ensure generated SAA map exists in 'policy_SAA/{config_name}/'.")
             
        # Copy to Result Dir
        if save_path:
             import shutil
             try:
                 shutil.copy(csv_path, os.path.join(save_path, "SAA_policy_map.csv"))
             except: pass
        
        # 1. Load Raw Map & Determine Max Dimensions
        print(f"  [SAA] Loading & Denseifying Policy Map from {csv_path}...")
        raw_map = {}
        max_indices = [0] * num_retailers
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                try:
                    import ast
                    key = ast.literal_eval(row[0])
                    if isinstance(key, int): key = (key,)
                    else: key = tuple(int(k) for k in key)
                    
                    # [Fix] Strict Mode: Always use row[2] (Value)
                    # SAA Map Format: [Key, Sample_Count, Value]
                    val = float(row[2])
                    
                    raw_map[key] = val
                    
                    # Track Max Index per Dimension
                    for i, k_val in enumerate(key):
                        if k_val > max_indices[i]:
                            max_indices[i] = k_val
                except: pass
        
        # [Fix] Enforce Symmetry for Homogeneous
        if self.is_homo:
            global_max = max(max_indices)
            max_indices = [global_max] * self.num_retailers
            print(f"  [SAA] Homo Symmetry Enforced: Set all dims to Max K={global_max}")

        # 2. Create Dense Arrays
        # Shape: Tailored to each retailer's max index + 2
        
        self.dim_sizes = tuple([m + 2 for m in max_indices])
        shape = self.dim_sizes
        
        print(f"  [SAA] Creating Dense Policy Table: Shape {shape}")
        
        self.policy_table = np.zeros(shape, dtype=np.float32)
        self.source_table = np.zeros(shape, dtype=np.int8) # 0: None, 1: Original, 2: Fallback
        
        # 3. Fill Originals
        count_orig = 0
        for k, v in raw_map.items():
            if all(x < self.dim_sizes[i] for i, x in enumerate(k)):
                self.policy_table[k] = v
                self.source_table[k] = 1 # Original
                count_orig += 1

        # 4. Fill Gaps (Pre-compute Fallback) for the ENTIRE table
        print("  [SAA] Pre-filling missing states via fallback logic...")
        
        it = np.nditer(self.policy_table, flags=['multi_index'])
        count_fallback = 0
        
        for _ in it:
            idx = it.multi_index
            if self.source_table[idx] != 0: continue
            
            # Homogeneous Symmetry Check
            filled_by_symmetry = False
            if not self.is_hetero:
                 s_idx = tuple(sorted(idx))
                 if s_idx != idx and self.source_table[s_idx] != 0:
                     self.policy_table[idx] = self.policy_table[s_idx]
                     self.source_table[idx] = 2
                     filled_by_symmetry = True
            
            if filled_by_symmetry:
                count_fallback += 1
                continue

            # Recursive Fallback
            prev_idx = tuple(max(0, x - 1) for x in idx)
            
            if idx == prev_idx:
                val = 0.0
            else:
                val = self.policy_table[prev_idx]
                
            self.policy_table[idx] = val
            self.source_table[idx] = 2 
            count_fallback += 1
            
        print(f"  [SAA] Setup Complete. Original: {count_orig}, Fallback-Filled: {count_fallback}")


    def get_action(self, obs, info):
        # O(1) Lookup
        if 'retailer_states' not in info:
             raise ValueError("SAA Need retailer_states")
             
        r_states = info['retailer_states']
        
        # 1. Prepare Key
        if not self.is_hetero:
            key = tuple(sorted(r_states))
        else:
            key = tuple(r_states)
            
        # 2. Key Validation / Clipping
        # If state exceeds table size, we clip to Max-1?
        # Or we rely on Pre-fill being enough?
        # If simulation goes to K=100 and table is size 20...
        # We must clip index to dim_size - 1.
        # This acts as answering "Max Known State" for any higher state.
        
        final_key = []
        for i, x in enumerate(key):
            ix = int(x)
            dim_lim = self.dim_sizes[i]
            if ix >= dim_lim: ix = dim_lim - 1
            if ix < 0: ix = 0
            final_key.append(ix)
        
        final_key = tuple(final_key)
        
        # 3. Lookup
        
        final_key = tuple(final_key)
        
        # 3. Lookup
        target_s = self.policy_table[final_key]
        
        # Check if it was a fallback (for tracking)
        is_fallback = (self.source_table[final_key] == 2)
        
        if 'm_IP' in info:
            m_ip = info['m_IP']
            return max(0.0, target_s - m_ip), is_fallback
        else:
             return 0.0, False

    def save_updated_map(self):
        # Save cache/fallback data to CSV
        if not self.save_path: return
        
        out_path = os.path.join(self.save_path, "SAA_policy_map_updated.csv")
        try:
            with open(out_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["Retailer_State_Pair", "Action_Target", "Source"])
                
                # Iterate table
                it = np.nditer(self.policy_table, flags=['multi_index'])
                for x in it:
                    idx = it.multi_index
                    src = self.source_table[idx]
                    src_str = "Original" if src == 1 else "Fallback"
                    if src == 0: continue # Should not happen after fill
                    
                    w.writerow([list(idx), f"{x:.2f}", src_str])
            print(f"  [SAA] Saving Updated Map with Fallbacks to {out_path}")
        except Exception as e:
            print(f"  [SAA] Failed to save updated map: {e}")


from env.vector_env import SubprocVecEnv

def make_env(config, seed, is_discrete):
    def _thunk():
        np.random.seed(seed)
        retailer_objects = []
        for r_conf in config['retailers']:
            delta = r_conf.get('delta', 0)
            mean_demand = r_conf.get('mean', 5)
            if is_discrete:
                customer_dist = poisson(mu=mean_demand)
            else:
                std_demand = r_conf.get('std', mean_demand ** 0.5) 
                customer_dist = HybridTruncNorm(mean=mean_demand, std=std_demand, lower=0)
            
            # [Refactor] Use dict instead of Retailer object
            retailer_objects.append({'delta': delta, 'customer': customer_dist})
            
        if is_discrete:
            env = EnvOWMRDiscreteAction(config, retailers=retailer_objects)
        else:
            env = EnvOWMRContinuous(config, retailers=retailer_objects)
        return env
    return _thunk

def run_numerical_experiment(args):
    # 1. Setup Base Paths
    # 1. Setup Base Paths
    if args.mode == "discrete":
        model_base_dir = f"saved_model_D"
        result_base_dir = f"result_D"
    elif args.mode == "continuous":
        model_base_dir = f"saved_model_C"
        result_base_dir = f"result_C"
    else:
        print("[Error] Mode must be 'discrete' or 'continuous'.")
        sys.exit(1)

    # [New] Override result dir for SAA
    if args.policy == 'saa':
        result_base_dir = "result_SAA"
    
        
    os.makedirs(result_base_dir, exist_ok=True)
    
    # Summary CSV Setup
    summary_path = os.path.join(result_base_dir, "summary.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Config", "LeadTime", "NumRetailers", "Heterogeneity", "Avg_Cost_Per_Step", "CI_95_Lower", "CI_95_Upper"])
    
    filters = getattr(args, 'filter', None)
    selects = getattr(args, 'select', None)
    experiments = get_experiment_list(args.mode, filters=filters, selects=selects)
    print(f"Found {len(experiments)} experiments.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_envs = args.num_envs if args.num_envs else 16
    
    processed_saa_groups = set()

    for idx_exp, exp in enumerate(experiments):
        config_name = exp['config']
        scenario_base = exp['scenario']
        is_discrete = exp['is_discrete']
        
        # Deduplicate SAA
        if args.policy == 'saa':
            group_id = get_saa_group_id(config_name)
            if group_id in processed_saa_groups:
                print(f"[{idx_exp+1}/{len(experiments)}] Skipping {config_name} (Duplicate SAA Group: {group_id})")
                continue
            processed_saa_groups.add(group_id)
            display_name = group_id
        else:
            display_name = config_name

        print(f"\n[{idx_exp+1}/{len(experiments)}] Processing {display_name} (Ref: {config_name})...")
        
        try:
            config = load_config(config_name)
        except Exception as e:
            print(f"  [Skip] Config Error: {e}")
            continue

        lead_time = config.get("lead_time", 0)
        hetero_type, num_retailers_parsed = parse_config_info(config_name)
        real_num_retailers = len(config.get("retailers", [])) 
        config['max_eps_length'] = 1000000 
        
        # [Fix] Inject config_name for SAA map loading in Env
        config['config_name'] = config_name 
        
        # [User Request] Skip Normalization for SAA Policy
        # SAA Policy uses raw values and does not require neural network state normalization.
        if args.policy == 'saa':
            config['skip_normalization'] = True
        else:
            config['skip_normalization'] = False 
        
        # Prepare Output Directory
        if args.policy == 'saa':
             save_path = os.path.join(result_base_dir, group_id)
        else:
             save_path = os.path.join(result_base_dir, config_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Policy Setup
        agent = None
        saa_policy = None
        
        if args.policy == 'saa':
            try:
                saa_policy = SAAPolicy(config_name, real_num_retailers, is_discrete, save_path=save_path)
            except Exception as e:
                print(f"  [Skip] SAA Error: {e}")
                continue
        else: # DRL Policy
            model_path = os.path.join(model_base_dir, config_name, "best_model.pt")
            if not os.path.exists(model_path):
                print(f"  [Skip] No model found at {model_path}")
                continue
                
            # Need to get dimensions from a temp env or config?
            # [Fix] Inject config_name for SAA map loading in Env
            config['config_name'] = config_name
            
            # Create a dummy env to get dimensions
            temp_env = make_env(config, 0, is_discrete)()
            state_dim = temp_env.getStateSize()
            action_dim = temp_env.getActionSize()
            
            agent = PPOMLPModel(config, state_dim, action_dim).to(device)
            try:
                agent.load_state_dict(torch.load(model_path, map_location=device))
            except Exception as e:
                print(f"  [Error] Load Model Failed: {e}")
                continue
            agent.eval()

        # Load Scenarios
        scenario_dir = os.path.join("numerical_scenario", scenario_base)
        scenario_files = sorted(glob.glob(os.path.join(scenario_dir, "*.npy")))
        if not scenario_files:
            print(f"  [Warning] No scenarios in {scenario_dir}")
            continue
            
        if not scenario_files:
            print(f"  [Warning] No scenarios in {scenario_dir}")
            continue
            
        if args.policy == 'saa':
             res_log_path = os.path.join(save_path, "result_log.csv")
        else:
             res_log_path = os.path.join(save_path, f"result_log.csv")
        # detailed_log_path removed
        
        # Initialize Vector Env
        
        
        # [Moved] Get Env consts for SAA normalization using a temporary single env
        if args.policy == 'saa':
            temp_env = make_env(config, seed=10000, is_discrete=is_discrete)()
            max_inv_val = temp_env.max_inventory
            temp_env.close()
        
        all_costs = []
        all_missed_lookups = [] # [New] Store total misses per scenario
        total_processed = 0
        
        # Chunk Processing
        scenario_chunks = [scenario_files[i:i + num_envs] for i in range(0, len(scenario_files), num_envs)]
        

        # Initialize Vector Env ONCE per Config (Reuse processes)
        # We assume seeds are handled by re-seeding or just continuing
        # But SubprocVecEnv requires make_env list.
        # Use dummy seeds for initializing list
        
        env_fns = [make_env(config, seed=10000 + i, is_discrete=is_discrete) for i in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns)

        try:
            for chunk_idx, chunk in enumerate(scenario_chunks):
                current_batch_size = len(chunk)
                
                # Prepare Demands
                batch_demands = []
                max_steps_in_batch = 0
                
                for s_path in chunk:
                    d = np.load(s_path)
                    batch_demands.append(d)
                    max_steps_in_batch = max(max_steps_in_batch, len(d))
                    
                # Pad Demands if batch < num_envs
                demand_args = []
                for i in range(num_envs):
                    if i < current_batch_size:
                        demand_args.append((batch_demands[i],))
                    else:
                        # Dummy demand (use first valid one or zeros)
                        # Env set_fixed_demands expects an array
                        if batch_demands:
                             dummy_d = np.zeros_like(batch_demands[0])
                        else:
                             dummy_d = np.zeros(10)
                        demand_args.append((dummy_d,))

                # Inject Demands (All Envs)
                vec_env.call_method_batch('set_fixed_demands', demand_args)
                
                # Reset
                obs = vec_env.reset()
                
                # Tracking
                batch_costs = np.zeros(current_batch_size)
                batch_steps = np.zeros(current_batch_size)
                
                # SAA Tracking
                missed_lookups_in_batch = np.zeros(current_batch_size, dtype=int)
                
                # Loop steps
                # We run for max_steps_in_batch. 
                
                for step in tqdm(range(max_steps_in_batch), desc=f"  [Chunk {chunk_idx+1}] Simulating", leave=False):
                    # Get Action
                    if args.policy == 'saa':
                        # Vectorize SAA
                        
                        if step == 0:
                            # Step 0: Fetch directly
                            m_ips = vec_env.get_attr('m_IP')
                            r_ks = vec_env.get_attr('r_k_states')
                        else:
                            # Step > 0: Use info
                            m_ips = [inf['m_IP'] for inf in infos]
                            r_ks = [inf['retailer_states'] for inf in infos]
                            
                        # Only take relevant ones? No, getting attrs returns all. 
                        # We just slice later.
                        m_ips = np.array(m_ips)
                        # r_ks might be list of lists
                        
                        # SAA Policy
                        actions_saa = []
                        # Track batch-wise missed lookups
                        batch_miss_counts = np.zeros(num_envs, dtype=int)
                        
                        # We must compute for all envs to keep VecEnv in sync?
                        # Using num_envs loop
                        for i in range(num_envs):
                            s_info = {
                                'retailer_states': r_ks[i],
                                'm_IP': m_ips[i]
                            }
                            # Lazy Lookup happens here
                            act, fallback = saa_policy.get_action(None, s_info)
                            actions_saa.append(act)
                            if fallback and i < current_batch_size:
                                batch_miss_counts[i] = 1
                        
                        actions_saa = np.array(actions_saa)
                        
                        # Accumulate misses (Only valid ones)
                        missed_lookups_in_batch += batch_miss_counts[:current_batch_size]
                        
                        # Target Level = Q + IP
                        target_s = actions_saa + m_ips 
                        
                        # Env expects Target Level
                        actions = target_s[:, np.newaxis]
                        
                        # Logging? We lost the logging variables from before removal, 
                        # but we still need 'actions' to step.

                    else:
                        # DRL (Residual Gaussian)
                        with torch.no_grad():
                            obs_tensor = torch.FloatTensor(obs).to(device)
                            ((mean, log_std), _) = agent(obs_tensor)
                            
                            # For evaluation/numerical experiment, we typically use the deterministic path.
                            # The "Mean" output represents the "Correction" amount.
                            # Action Logic: Target = SAA - Correction
                            
                            correction = mean
                            
                            saa_base = agent.get_saa_base_action(obs_tensor)
                            
                            # Logic: Target = SAA - Correction
                            # Clamp SAA - Correction between [0, SAA]
                            # This means Correction must be effectively in [0, SAA]
                            
                            
                            raw_target = saa_base - correction
                            
                            final_action = torch.clamp(raw_target, min=0.0)
                            # final_action = torch.min(final_action, saa_base) <-- REMOVED
                            
                            actions = final_action.cpu().numpy() # (N, 1)
                    
                    # Step
                    obs, rewards, dones, infos = vec_env.step(actions)
                    
                    # Accumulate Costs (Only for valid batch indices)
                    # Helper for slicing assumes rewards is (N,) or (N,1)
                    r_valid = rewards[:current_batch_size]
                    current_costs = -r_valid.flatten() * 100.0
                    
                    batch_costs += current_costs
                    batch_steps += 1
                    
                # End of Chunk Loop -> Collect
                for i in range(current_batch_size):
                    if batch_steps[i] > 0:
                        # Note: If scenarios have different lengths, we might be adding zeros or skipping?
                        # We assume max_steps_in_batch is correct.
                        # If a scenario finished earlier, EnvOWMR usually stops producing cost?
                        # Or returns 0?
                        # We divide by steps run.
                        avg_cost = batch_costs[i] / batch_steps[i]
                        all_costs.append(avg_cost)
                        if args.policy == 'saa':
                            all_missed_lookups.append(missed_lookups_in_batch[i])
        
        finally:
            # Close VecEnv once after all chunks
            vec_env.close()

        # Save Update SAA Map (Once at end)
        if args.policy == 'saa' and hasattr(saa_policy, 'save_updated_map'):
            saa_policy.save_updated_map()
        # detailed_file.close() removed

        # Stats
        mean_cost = np.mean(all_costs)
        std_cost = np.std(all_costs)
        n = len(all_costs)
        ci_lower, ci_upper = calculate_ci(mean_cost, std_cost, n)
        
        # Result Log
        with open(res_log_path, 'w', newline='') as f:
            w = csv.writer(f)
            if args.policy == 'saa':
                w.writerow(["Scenario_ID", "Avg_Cost_Per_Step", "Missed_SAA_Lookups"])
                for i, c in enumerate(all_costs):
                    w.writerow([i, c, all_missed_lookups[i]])
            else:
                w.writerow(["Scenario_ID", "Avg_Cost_Per_Step"])
                for i, c in enumerate(all_costs):
                    w.writerow([i, c])
            w.writerow([])
            w.writerow(["Metric", "Value"])
            w.writerow(["Total_Avg_Cost", mean_cost])
            w.writerow(["Std_Dev", std_cost])
            w.writerow(["CI_95_Lower", ci_lower])
            w.writerow(["CI_95_Upper", ci_upper])
            
        # Summary Append
        try:
            with open(summary_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([config_name, lead_time, real_num_retailers, hetero_type, f"{mean_cost:.2f}", f"{ci_lower:.2f}", f"{ci_upper:.2f}"])
        except Exception as e:
            print(f"  [Error] Failed to write to summary.csv: {e}")
            
        # Policy Map
        if args.policy != 'saa':
            try:
                # Need dummy env for plotting?
                # Create one temporary env
                dummy_env = make_env(config, 0, is_discrete)()
                generate_full_policy_map(dummy_env, agent, device, config_name, save_path, is_discrete)
                dummy_env.close()
            except Exception as e:
                 print(f"  [Warning] Plotting failed: {e}")
                 
        print(f"  [Config Done] Avg Cost: {mean_cost:.2f} (Scenarios: {len(all_costs)})")

    print("\n[Done] Numerical Experiments Completed.")

if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["discrete", "continuous"])
    parser.add_argument("--filter", nargs='+', default=None, help="Substring match configs")
    parser.add_argument("--select", nargs='+', default=None, help="Exact match configs")
    parser.add_argument("--policy", type=str, default="drl", choices=["drl", "saa"])
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel Batch Size")
    
    args = parser.parse_args()
    
    run_numerical_experiment(args)
# python numerical_experiment.py --mode continuous --filter Homo_5R_0_0_0 Homo_5R_0_0_1 Homo_5R_0_0_2 Homo_5R_0_0_3