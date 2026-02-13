import argparse
import os
import numpy as np
from scipy.stats import poisson, norm, truncnorm
import importlib
import sys

def generate_scenarios(args):
    """
    Generate fixed demand scenarios based on a Config file.
    """
    
    # 1. Output Naming & Duplicate Check
    if args.name:
        folder_name = args.name
    elif args.config:
        parts = args.config.split('_')
        # Recursively strip numeric suffixes, but keep at least 2 parts (e.g. Homo_5R)
        # to share scenarios across different Lead Times.
        stripped = False
        while len(parts) > 2 and parts[-1].isdigit():
            parts.pop()
            stripped = True
            
        if stripped:
            folder_name = "_".join(parts)
            print(f"[Info] Consolidating output for '{args.config}' into '{folder_name}'")
        else:
            folder_name = args.config
    else:
        print("[Error] Must provide --config (and optionally --name).")
        sys.exit(1)
        
    base_path = "evaluate"
    save_path = os.path.join(base_path, folder_name)
    
    if os.path.exists(save_path) and not args.force:
        print(f"[Info] Simulation folder '{save_path}' already exists.")
        print("       Skipping generation to prevent duplicates.")
        print("       (Use --force to overwrite)")
        sys.exit(0)

    os.makedirs(save_path, exist_ok=True)
    
    # 2. Config Validation
    if not args.config:
         print("[Error] --config argument is required.")
         sys.exit(1)
    
    retailer_params = []
    num_retailers = 0
    
    # 3. Load Config
    # Add current directory to path for config loading
    sys.path.append(os.getcwd())

    try:
        print(f"Loading settings from config: {args.config}")
        # Dynamic import: config.Homo_2R_6
        config_module = importlib.import_module(f"config.{args.config}")
        raw_config = config_module.config
        
        retailers_conf = raw_config['retailers']
        num_retailers = len(retailers_conf)
        
        for r_conf in retailers_conf:
            # Extract mean/std. 
            r_mean = r_conf.get('mean', 5.0)
            r_std = r_conf.get('std', r_mean * 0.2)
            
            # Logic: If 'std' is in config -> trunc_normal
            # If 'std' is NOT in config -> poisson
            
            # [LOGIC UPDATE] Force Poisson for Discrete configurations (D_*)
            if args.config.startswith("D_"):
                 dist_type = "poisson"
            elif 'std' in r_conf:
                dist_type = "trunc_normal"
            
            retailer_params.append({
                "mean": r_mean,
                "std": r_std,
                "dist": dist_type
            })
            
    except Exception as e:
        print(f"[Error] Failed to load config '{args.config}': {e}")
        print("       Make sure the config file exists in 'config_con/' directory.")
        sys.exit(1)

    # 4. Generate Data
    print(f"Generating {args.folders} scenarios...")
    print(f" Output: {save_path}")
    print(f" Num Retailers: {num_retailers}")
    print(f" Steps per Scenario: {args.len}")

    rng = np.random.default_rng(seed=100)

    for i in range(args.folders):
        scenario_columns = []
        
        for r_idx in range(num_retailers):
            params = retailer_params[r_idx]
            mu = params["mean"]
            sigma = params["std"]
            current_dist = params["dist"]
            
            if current_dist == "poisson":
                # Poisson takes 'mu'
                data = poisson.rvs(mu=mu, size=args.len, random_state=rng)
            elif current_dist == "trunc_normal":
                # args.lower defaults to 0
                lower = args.lower
                # Standardize
                a = (lower - mu) / sigma
                b = np.inf
                data = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=args.len, random_state=rng)

            scenario_columns.append(data)
            
        # Stack -> (Steps, Retailers)
        scenario_array = np.array(scenario_columns).T
        
        # Save
        np.save(os.path.join(save_path, f"scenario_{i}.npy"), scenario_array)

    print("Generation Complete.")

def run_batch_generation(args):
    """
    Find all config files and run generation for unique scenario groups.
    """
    config_dir = os.path.join(os.getcwd(), 'config')
    files = [f for f in os.listdir(config_dir) if f.endswith('.py') and f != '__init__.py']
    
    # Filter by mode (start with D_ for discrete, else continuous)
    # Default to continuous for now unless specified otherwise in args logic?
    # Actually, let's just do ALL consistent with SAA logic.
    # But usually we want to separate Continuous vs Discrete runs.
    # Let's just process ALL files found.
    
    # Group by base name (e.g. Homo_1R) to avoid duplicate work
    groups = {}
    for f in files:
        c_name = f[:-3] # remove .py
        
        # Determine Base (Scenario Group)
        parts = c_name.split('_')
        # Logic: Strip numeric tail until we hit the identifier part
        # E.g. Homo_1R_4_2 -> Homo_1R
        # But wait, Homo_1R_0 and Homo_1R_2 share same retailers? Yes.
        # So we want to group by Retailer Config.
        
        # Simplified grouping: 
        # Homo_1R_*, Homo_2R_*, Hetero_2R_*, Homo_5R_*
        # Logic: Take first 2 parts? Homo_1R
        
        if c_name.startswith("D_"):
            # D_Homo_1R_...
            group_key = "_".join(parts[:3]) # D_Homo_1R
        else:
            group_key = "_".join(parts[:2]) # Homo_1R
            
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(c_name)
        
    print(f"Found {len(files)} configs. Grouped into {len(groups)} unique scenario sets.")
    
    for g_key, c_list in groups.items():
        # Pick the first config in the group as representative
        rep_config = c_list[0]
        print(f"\n[Batch] Processing Group: {g_key} (Example: {rep_config})")
        
        # Construct fake args for the single run
        # We need to preserve other flags like --force, --len, etc.
        # Create a new Namespace or modify current args
        
        # We must set args.config to the representative config
        args.config = rep_config
        # We must set args.name to the group key to ensure unified output folder
        args.name = g_key
        
        try:
             generate_scenarios(args)
        except Exception as e:
             print(f"  [Error] Failed to generate for {g_key}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Fixed Demand Scenarios via Config")
    
    # 1. Output/Input naming
    parser.add_argument("--config", type=str, default=None, help="Config file name (e.g. Hetero_2R_6)")
    parser.add_argument("--name", type=str, default=None, help="Output folder name (e.g. Homo_2R)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing folder")
    parser.add_argument("--all", action="store_true", help="Run for ALL config groups in config/ folder")

    # 2. Generator settings
    parser.add_argument("--lower", type=float, default=0.0, help="Lower bound for TruncNormal (default: 0.0)")
    parser.add_argument("--len", type=int, default=10000, help="Steps per scenario")
    parser.add_argument("--folders", type=int, default=10, help="Number of scenarios")

    args = parser.parse_args()
    
    if args.all:
        run_batch_generation(args)
    elif args.config:
        generate_scenarios(args)
    else:
        parser.print_help()
