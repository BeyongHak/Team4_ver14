
import argparse
import os
import numpy as np
import glob
import importlib
import sys
from scipy.stats import poisson, truncnorm

def generate_numerical_scenarios(args):
    """
    Generate fixed demand scenarios for all configs in 'config/' folder.
    Saves to 'numerical_scenario/[ConfigName]/'.
    Uses a distinct base seed (default 20000).
    """
    
    # 1. Base Settings
    base_save_path = "numerical_scenario"
    config_dir = "config"
    pattern = "*.py"
    
    # 2. Add current directory to path
    sys.path.append(os.getcwd())
    
    # 3. Discover Configs
    files = sorted(glob.glob(os.path.join(config_dir, pattern)))
    
    print(f"found {len(files)} config files in '{config_dir}'")
    
    # 4. Group Configs by Scenario Name
    # We only need one set of scenarios per unique retailer configuration (Name without Lead Time)
    # e.g., Homo_2R_0, Homo_2R_2 -> Both map to "Homo_2R" (or "D_Homo_2R")
    
    unique_scenarios = {} # {scenario_name: config_path}
    
    for f in files:
        filename = os.path.basename(f)
        if filename == "__init__.py":
            continue
            
        config_name = filename.replace(".py", "")
        
        # Determine Scenario Base Name
        # Rule: Remove ALL trailing parts if they are digits (Lead Time, version, etc.)
        parts = config_name.split('_')
        while parts and parts[-1].isdigit():
            parts.pop()
            
        scenario_name = "_".join(parts)
            
        if scenario_name not in unique_scenarios:
            unique_scenarios[scenario_name] = config_name
            
    print(f"found {len(files)} config files, reduced to {len(unique_scenarios)} unique scenario groups.")
    
    # Filter Scenarios if requested
    if args.filter:
        print(f"Filtering scenarios with keywords: {args.filter}")
        filtered_scenarios = {}
        for name, cfg in unique_scenarios.items():
            if any(k in name for k in args.filter):
                filtered_scenarios[name] = cfg
        unique_scenarios = filtered_scenarios
        print(f"-> Selected {len(unique_scenarios)} scenario groups after filtering.")
    
    for scenario_name, config_name in unique_scenarios.items():
        
        # Load Config
        try:
            config_module = importlib.import_module(f"config.{config_name}")
            raw_config = config_module.config
        except Exception as e:
            print(f"[Error] Failed to load {config_name}: {e}")
            continue

        # Setup Save Path -> numerical_scenario/[ScenarioName]
        # NOT [ConfigName]
        save_path = os.path.join(base_save_path, scenario_name)
        
        if os.path.exists(save_path) and not args.force:
            print(f"[Skip] {scenario_name} - Folder exists (Use --force to overwrite)")
            continue
            
        os.makedirs(save_path, exist_ok=True)
        
        # Retailer Params
        retailers_conf = raw_config['retailers']
        num_retailers = len(retailers_conf)
        retailer_params = []
        
        # [Strict Logic] Determine distribution based on Config Name
        # If starts with 'D_', it is Discrete (Poisson)
        # Else, it is Continuous (TruncNormal with supplychain.py logic)
        is_discrete_config = config_name.startswith("D_")
        
        for r_conf in retailers_conf:
            r_mean = r_conf.get('mean', 5.0)
            
            # For Continuous: Enforce supplychain.py logic (std = sqrt(mean))
            r_std = r_mean ** 0.5
            
            if is_discrete_config:
                dist_type = "poisson"
            else:
                dist_type = "trunc_normal"
            
            retailer_params.append({
                "mean": r_mean,
                "std": r_std,
                "dist": dist_type
            })
            
        # Generate Scenarios
        print(f"[Generating] {scenario_name} (Type: {dist_type}) -> {save_path}")
        
        # Base Seed for this batch
        base_seed = 20000 
        rng = np.random.default_rng(seed=base_seed)

        for i in range(args.folders):
            scenario_columns = []
            
            for r_idx in range(num_retailers):
                params = retailer_params[r_idx]
                mu = params["mean"]
                sigma = params["std"]
                current_dist = params["dist"]
                
                if current_dist == "poisson":
                    # Discrete: Poisson
                    data = poisson.rvs(mu=mu, size=args.len, random_state=rng)
                elif current_dist == "trunc_normal":
                    # Continuous: Truncated Normal (supplychain.py logic)
                    # a = (lower - mean) / std, b = inf
                    lower = args.lower
                    a = (lower - mu) / sigma
                    b = np.inf
                    data = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=args.len, random_state=rng)
                else:
                    # Fallback
                    data = poisson.rvs(mu=mu, size=args.len, random_state=rng)
                
                scenario_columns.append(data)
                
            # Stack -> (Steps, Retailers)
            scenario_array = np.array(scenario_columns).T
            
            # Save
            np.save(os.path.join(save_path, f"scenario_{i}.npy"), scenario_array)

    print("\nAll Numerical Scenarios Generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Numerical Experiment Scenarios")
    
    parser.add_argument("--force", action="store_true", help="Overwrite existing folders")
    parser.add_argument("--lower", type=float, default=0.0, help="Lower bound for TruncNormal (default: 0.0)")
    parser.add_argument("--len", type=int, default=10000, help="Steps per scenario (Default 10000 for verification)")
    parser.add_argument("--folders", type=int, default=96, help="Number of scenarios per config (Default 10)")
    parser.add_argument("--filter", nargs='*', help="Keywords to filter scenario names")

    args = parser.parse_args()
    
    generate_numerical_scenarios(args)

# python generate_numerical_scenarios.py --force --lower 0.0 --len 100000 --folders 100 --filter Homo_5R_0_0_0 Homo_5R_0_0_1 Homo_5R_0_0_2 Homo_5R_0_0_3

