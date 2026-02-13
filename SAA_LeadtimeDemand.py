
import os
import sys
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import importlib

from scipy.stats import poisson

# Add custom paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'env'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

# from env.retailer import Retailer <--- Removed

# Helper for loading config (reuse from main/numerical)
def load_config(config_name):
    config_module = importlib.import_module(f"config.{config_name}")
    return config_module.config

def parse_config_info(config_name):
    parts = config_name.split('_')
    # Example: Homo_5R_8_0
    hetero_type = parts[0] # Homo or Hetero
    num_retailers = 0
    for p in parts:
        if p.endswith('R'):
             try:
                 num_retailers = int(p[:-1])
                 break
             except: 
                 pass
    return hetero_type, num_retailers




def get_scenario_base_name(config_name):
    """
    Robust logic to extract base scenario name from config name.
    Example: Homo_5R_8_0_0 -> Homo_5R_8
             Hetero_2R_4_1 -> Hetero_2R_4
    Logic: Strip trailing numeric parts, but preserve at least 3 parts if they exist (Type_NumR_LeadTime).
    """
    parts = config_name.split('_')
    
    # Heuristic: Keep stripping last part if it is digits, 
    # BUT stop if we only have 3 parts left (e.g. Homo_5R_8 should stay Homo_5R_8, not Homo_5R)
    # Actually, LeadTime is usually the 3rd part. 
    # Let's ensure we don't strip the Lead Time.
    
    temp_parts = list(parts)
    
    # [Mod] Handle Prefix 'D_' (Discrete)
    # If D_, we want D_Type_NumR_LeadTime (4 parts)
    # Else, we want Type_NumR_LeadTime (3 parts)
    
    min_len = 4 if parts[0] == 'D' else 3
    
    while len(temp_parts) > min_len and temp_parts[-1].isdigit():
        temp_parts.pop()
        
    return "_".join(temp_parts)

def calculate_saa_policy(config_name_or_args, sim_steps=1000000, output_name=None):
    """
    Main Logic for SAA Leadtime Demand Calculation
    """
    if isinstance(config_name_or_args, str):
        config_name = config_name_or_args
    else:
        config_name = config_name_or_args.config
        if hasattr(config_name_or_args, 'steps'):
            sim_steps = config_name_or_args.steps

    # Auto-infer grouping name if output_name not provided
    if output_name is None:
        output_name = get_scenario_base_name(config_name)

    # 1. Output Setup
    folder_name = output_name 
    output_dir = "policy_SAA"
    os.makedirs(output_dir, exist_ok=True)
    
    final_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(final_output_dir, exist_ok=True)
    
    saa_output_path = final_output_dir
    
    print(f"[{config_name}] Starting SAA Analysis...")
    print(f"  Target Group: {folder_name}")
    print(f"  Output Directory: {saa_output_path}")
    
    try:
        config = load_config(config_name)
    except ImportError:
        print(f"Error: Config {config_name} not found.")
        return



    # 2. Key Parameters
    # [Modified] Auto-detect mode from config name (Robustness)
    is_discrete = config_name.startswith("D_")
    if is_discrete:
        print(f"  [Mode] Detected Discrete Config (Prefix 'D_')")
    else:
        print(f"  [Mode] Detected Continuous Config (No 'D_' prefix)")
    lead_time = config.get("lead_time", 2)
    holding_cost = config.get("holding_cost", 1.0)
    backorder_cost = config.get("backorder_cost", 4.0) 
    
    target_fractile = backorder_cost / (holding_cost + backorder_cost)
    print(f"  Cost Ratio (b/b+h): {target_fractile:.4f} (h={holding_cost}, b={backorder_cost})")
    print(f"  Lead Time (L): {lead_time} (Demand Window: L+1 = {lead_time+1})")
    
    hetero_type, _ = parse_config_info(config_name)
    is_homo = (hetero_type == "Homo")
    if is_homo:
        print("  Type: Homogeneous (Will sort states for aggregation)")
    else:
        print("  Type: Heterogeneous (Strict state pairing)")

    # 3. Setup Environment & Retailers
    # 3. Setup Environment & Retailers
    retailer_objects = []
    for r_conf in config['retailers']:
        delta = r_conf.get('delta', 150)
        mean_demand = r_conf.get('mean', 5)
        # Note: Discrete/Continuous specific distributions
        if is_discrete:
             # Standard Poisson
             from scipy.stats import poisson
             customer_dist = poisson(mu=mean_demand)
        else:
             from env.truncNormal import HybridTruncNorm
             std_demand = r_conf.get('std', mean_demand**0.5)
             # Should match env definition
             customer_dist = HybridTruncNorm(mean=mean_demand, std=std_demand, lower=0)
             
        # [Refactor] Use dict instead of Retailer object to avoid OverflowError in K calculation
        retailer_objects.append({'delta': delta, 'customer': customer_dist})
    num_retailers = len(retailer_objects)

    # [Modified] Vectorized SAA Simulation (Pure Random Mode)
    BATCH_SIZE = 10
    # [Modified] Interpret 'sim_steps' as 'Steps Per Env'
    steps_per_env = max(lead_time + 2, sim_steps)
    NUM_CHUNKS = 1 # Simplified to 1 big run for logic consistency
    
    total_samples = BATCH_SIZE * steps_per_env
    print(f"  [Mode] Pure Random Sampling. {BATCH_SIZE} Parallel Envs.")
    print(f"  Steps Per Env: {steps_per_env:,}")
    print(f"  Total Samples: {BATCH_SIZE:,} x {steps_per_env:,} = {total_samples:,} samples.")
    
    
    # Init State Variables (Batch, NumRetailers)
    # Start with full inventory
    r_IP = np.zeros((BATCH_SIZE, num_retailers))
    r_k_states = np.zeros((BATCH_SIZE, num_retailers), dtype=int)
    
    for i, r in enumerate(retailer_objects):
        r_IP[:, i] = r['delta']
        r_k_states[:, i] = 0
        
    # LTD Calculation Buffer: (Batch, WindowSize, NumRetailers)
    window_size = lead_time + 1
    
    # Init Buffers for Lookahead
    # Buffer to store states: (Batch, Window, NumR)
    snapshot_len = window_size
    state_buffer_k = np.zeros((BATCH_SIZE, snapshot_len, num_retailers), dtype=int)
    # Buffer to store system orders: (Batch, Window)
    order_buffer_val = np.zeros((BATCH_SIZE, snapshot_len))
    
    chunk_states = [] 
    chunk_ltds = []   
    
    # [Optimized] Aggregate into dictionary incrementally to save memory
    saa_data = {}
    FLUSH_INTERVAL = 50000 # Flush every 50k batch steps (50k * 1000 = 50M samples)
    MAX_SAMPLES_PER_STATE = 10000 # [New] Memory Cap: Stop collecting after 10k samples per state
    
    # Helper to flush buffer to dictionary
    def flush_buffer(c_states, c_ltds, data_dict):
        if not c_states: return
        
        # Stack
        batch_stack = np.vstack(c_states) # (N, R)
        ltd_stack = np.concatenate(c_ltds) # (N,)
        
        # Insert using loop
        for idx in range(len(ltd_stack)):
            s_vec = batch_stack[idx]
            val = ltd_stack[idx]
            
            if is_homo:
                s_vec.sort()
                key = tuple(s_vec)
            else:
                key = tuple(s_vec)
                
            # [Modified] Sample Capping Logic REMOVED per user request
            if key not in data_dict:
                data_dict[key] = []
            
            data_dict[key].append(val)
            # Cap removed. All samples are collected.
    
    # Detailed Log (First Batch Only, First 50 steps)
    detailed_log_limit = 50
    detailed_records = []

    for t in tqdm(range(steps_per_env), desc="Vectorized Sim"):
        
        # 1. Capture Decision State (Before Order)
        current_k = r_k_states.copy() # (Batch, NumR)
        
        # 2. Retailer Logic (Order & Update)
        orders = np.zeros((BATCH_SIZE, num_retailers))
        
        for i, r in enumerate(retailer_objects):
            mask_reorder = r_IP[:, i] <= 0
            # Order
            orders[mask_reorder, i] = r['delta'] - r_IP[mask_reorder, i]
            # Reset IP
            r_IP[mask_reorder, i] = r['delta']
            # Update State
            r_k_states[mask_reorder, i] = 0
            r_k_states[~mask_reorder, i] += 1
            
        # 3. Demand Realization & IP Update
        # Random Generation
        # [Optimization] Generate in bulk? No, memory usage.
        demands = np.zeros((BATCH_SIZE, num_retailers))
        for i, r in enumerate(retailer_objects):
            demands[:, i] = r['customer'].rvs(size=BATCH_SIZE)
                
        r_IP -= demands
        
        # 4. Update Rolling Buffers
        system_orders = np.sum(orders, axis=1) # (Batch,)
        
        # Shift Buffers (Left)
        state_buffer_k[:, :-1, :] = state_buffer_k[:, 1:, :]
        order_buffer_val[:, :-1] = order_buffer_val[:, 1:]
        
        # Push Newest (Right)
        state_buffer_k[:, -1, :] = current_k
        order_buffer_val[:, -1] = system_orders
        
        # 5. Extract Valid Samples (Only if buffer full)
        if t >= snapshot_len - 1:
            # Oldest State at index 0
            valid_state = state_buffer_k[:, 0, :] # (Batch, NumR)
            # Sum of Orders in buffer is LTD
            valid_ltd = np.sum(order_buffer_val, axis=1) # (Batch,)
            
            chunk_states.append(valid_state.copy())
            chunk_ltds.append(valid_ltd.copy())
            
        # [New] Incremental Flush
        if len(chunk_states) >= FLUSH_INTERVAL:
             # print(f"  [Memory] Flushing {len(chunk_states) * BATCH_SIZE} samples to dict...")
             flush_buffer(chunk_states, chunk_ltds, saa_data)
             chunk_states = []
             chunk_ltds = []
            
        # 6. Verification Log (Trace Batch 0)
        if t < detailed_log_limit:
            row_orders = orders[0]
            row_demands = demands[0]
            row_ips = r_IP[0]
            row_k = current_k[0]
            
            if t >= snapshot_len - 1:
                log_ltd = np.sum(order_buffer_val[0])
            else:
                log_ltd = 0.0
                
            detailed_records.append({
                "t": t,
                "State_k_vec": str(tuple([int(x) for x in row_k])),
                "Order_at_t": system_orders[0],
                "Calculated_LTD": log_ltd,
                "Retailer_Details": []
            })
            for r_idx in range(num_retailers):
                detailed_records[-1]["Retailer_Details"].append({
                    "State": row_k[r_idx],
                    "Demand": row_demands[r_idx],
                    "Order": row_orders[r_idx],
                    "IP_After": row_ips[r_idx]
                })

    # Final Flush
    if chunk_states:
         print(f"  [Memory] Final flush of {len(chunk_states) * BATCH_SIZE} samples...")
         flush_buffer(chunk_states, chunk_ltds, saa_data)

    # --- NEW LOGGING LOOP ---
    ver_path = os.path.join(saa_output_path, "SAA_verification.csv")
    print(f"  Writing verification log to {ver_path}...")
    with open(ver_path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ["Step", "State_k_vec", "Order_at_t", "Calculated_LTD"]
        for i in range(num_retailers):
            header.extend([f"R{i}_State", f"R{i}_Demand", f"R{i}_Order", f"R{i}_IP_After"])
        w.writerow(header)
        for record in detailed_records:
            row = [record["t"], record["State_k_vec"], f"{record['Order_at_t']:.2f}", f"{record['Calculated_LTD']:.2f}"]
            for r_dat in record["Retailer_Details"]:
                row.extend([r_dat["State"], f"{r_dat['Demand']:.2f}", f"{r_dat['Order']:.2f}", f"{r_dat['IP_After']:.2f}"])
            w.writerow(row)

    # End of Chunk Loop - Aggregate (Removed, already done in loop)
    # saa_data is now populated

    csv_path = os.path.join(saa_output_path, "SAA_policy_map.csv")
    
    print(f"  Calculating {target_fractile*100:.1f}th Percentile for {len(saa_data)} states...")
    
    
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Retailer_State_Pair", "Sample_Count", "Optimal_Order_Up_To_Level"])
        
        sorted_keys = sorted(saa_data.keys())
        for key in sorted_keys:
            samples = saa_data[key]
            # [Modified] SAA Logic: Direct Indexing (No Interpolation)
            # Find the smallest value x in samples such that P(D <= x) >= target_fractile
            # This corresponds to the sample at index ceil(N * fractile) - 1 in sorted list.
            
            samples.sort() # Ensure sorted
            n_samples = len(samples)
            target_idx = int(np.ceil(n_samples * target_fractile)) - 1
            target_idx = max(0, min(target_idx, n_samples - 1)) # Safety clamp
            
            optimal_x = samples[target_idx]
            
            # [Modified] Conditional Formatting
            # If Config starts with 'D_', it is discrete -> Use int (though sample is likely int already)
            # Else (Continuous) -> Use float
            is_discrete_config = config_name.startswith("D_")
            
            if is_discrete_config:
                final_val = int(optimal_x) # Value from sample is already discrete
            else:
                final_val = float(optimal_x)
            
            
            # Clean Key for CSV
            clean_key = [int(k) for k in key]
            # Use appropriate formatting
            if is_discrete_config:
                 w.writerow([clean_key, len(samples), final_val])
            else:
                 w.writerow([clean_key, len(samples), f"{final_val:.4f}"])
            
    print(f"  Policy Map saved to {csv_path}")
    print(f"  [Done] SAA Calculation Complete for {folder_name}.")

def run_batch_saa(args):
    # Find all config files
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    # Filter only .py files and exclude __init__
    all_files = [f for f in os.listdir(config_dir) if f.endswith('.py') and f != '__init__.py']
    
    # Smart Filtering based on Mode
    filtered_files = []
    if args.mode == 'discrete':
        filtered_files = [f for f in all_files if f.startswith('D_')]
        print("  [Smart Filter] Mode=Discrete -> Selected only 'D_*' configs.")
    else:
        filtered_files = [f for f in all_files if not f.startswith('D_')]
        print("  [Smart Filter] Mode=Continuous -> Excluded 'D_*' configs.")
        
    # User Keyword Filter (Select OR Filter)
    target_configs = []
    
    has_selection = False
    
    # 1. Select (Exact Match)
    if args.select:
        has_selection = True
        print(f"  [Select] Looking for exact matches: {args.select}")
        for f in filtered_files:
            c_name = f[:-3]
            if c_name in args.select:
                target_configs.append(c_name)

    # 2. Filter (Substring Match)
    if args.filter:
        has_selection = True
        print(f"  [Filter] Looking for substrings: {args.filter}")
        for f in filtered_files:
            c_name = f[:-3]
            # Check if ANY filter keyword is in the config name
            if any(k in c_name for k in args.filter):
                target_configs.append(c_name)

    # 3. Default (All)
    if not has_selection:
        print("  [Filter] No keywords provided. Running all mode-compatible configs.")
        target_configs = [f[:-3] for f in filtered_files]

    # Unique and Sorted
    target_configs = sorted(list(set(target_configs)))
    print(f"Found {len(target_configs)} configs to process: {target_configs}")

    # --- NEW GROUPING LOGIC ---
    # Group configs by Type_NumR_LeadTime
    groups = {} 
    for c_name in target_configs:
        group_id = get_scenario_base_name(c_name)
        
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(c_name)

    print(f"\nIdentified {len(groups)} unique groups for SAA calculation.")

    for group_id, members in groups.items():
        rep_config = members[0]
        print(f"\n[Group: {group_id}] Calculating SAA for representative: {rep_config}")
        
        try:
             # Just pass the config name. calculate_saa_policy will handle auto-discovery.
             calculate_saa_policy(rep_config, sim_steps=args.steps)
        except Exception as e:
            print(f"  [Error] Failed processing {rep_config}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", nargs='*', help="Keywords to filter config names (Substring match)")
    parser.add_argument("--select", nargs='*', help="Exact Config Names to run (Exact match)")
    parser.add_argument("--config", type=str, help="Single Config Name (Optional - direct run)")
    parser.add_argument("--mode", type=str, default="discrete", choices=["discrete", "continuous"])
    parser.add_argument("--steps", type=int, default=100000, help="Simulation steps per Environment (Total = 10 x Steps)")
    
    args = parser.parse_args()
    
    if args.config:
        # Single run
        # Need to handle scenario logic for single run too
        # Infer group name from config
        parts = args.config.split('_')
        # Simple heuristic
        group_id = args.config 
        # But wait, run_batch_saa handles grouping. 
        # For single run, just pass config.
        # Single run
        # Just pass config name; calculate_saa_policy handles scenario discovery.
        calculate_saa_policy(args.config, sim_steps=args.steps)
        
    elif args.filter is not None or args.select is not None:
        # Batch run
        run_batch_saa(args)
    else:
        parser.print_help()
