
import subprocess
import sys
import time
import os
import glob
import argparse

# ==============================================================================
# Global Settings
# ==============================================================================
MAX_UPDATES = 500000   # Reduced to 500K
EVAL_INTERVAL = 1000   # Evaluates every 5000 updates (Total ~100 evals)
PYTHON_EXE = sys.executable

def get_scenario_name(config_filename):
    """
    Infers scenario name from config filename.
    Ex: 'D_Homo_2R_6.py' -> 'D_Homo_2R'
    Ex: 'Homo_1R_4'      -> 'Homo_1R'
    Ex: 'Homo_.._ReLU'   -> 'Homo_..' (strips suffix)
    """
    name = config_filename.replace(".py", "")
    
    # [Fix] Strip known experimental suffixes
    if name.lower().endswith("_relu"):
        name = name[:-5]

    parts = name.split('_')
    
    # Recursively remove trailing digit parts (e.g., _8, _0)
    # Goal: 'Homo_5R_8_0_0' -> 'Homo_5R'
    # Fixed: Keep only first 2 parts (Type_NumR) if possible to share scenarios across LeadTimes
    while len(parts) > 2 and parts[-1].isdigit():
        parts.pop()
        
    return "_".join(parts)

def get_experiment_list(mode="all", whitelist=None):
    """
    Automatically discovers config files in config/
    Returns a list of dicts: {'config': 'Name', 'scenario': 'Name'}
    
    Arguments:
        mode (str): 'all', 'discrete', or 'continuous'
        whitelist (list): Optional list of strings.
                          Implements SMART FILTER:
                          - If a term exactly matches a config name, select ONLY that config.
                          - Otherwise, select all configs where term is a substring.
    """
    pattern = "*.py" 
    config_dir = "config"
    files = glob.glob(os.path.join(config_dir, pattern))
    
    # Custom Sort Function
    def sort_key(filename):
        # Remove extension and prefix for parsing
        name_clean = filename.replace(".py", "").replace("D_", "")
        parts = name_clean.split('_')
        
        # Default values
        num_retailers = 999
        is_hetero = 1 # 0 for Homo, 1 for Hetero
        lead_time = 0
        
        # Parse parts
        for p in parts:
            if "R" in p and p[:-1].isdigit(): # 1R, 2R
                num_retailers = int(p[:-1])
            if p == "Homo":
                is_hetero = 0
            if p == "Hetero":
                is_hetero = 1
                
        if parts[-1].isdigit():
            lead_time = int(parts[-1])
            
        return (num_retailers, is_hetero, lead_time)

    # Sort files based on custom key
    files = sorted(files, key=lambda f: sort_key(os.path.basename(f)))
    
    # 1. Collect Valid Candidates (based on Mode)
    candidates = []
    
    for f in files:
        filename = os.path.basename(f)
        if filename == "__init__.py":
            continue
            
        config_name = filename.replace(".py", "")
        
        is_discrete = config_name.startswith("D_")
        
        # Mode Filter
        if mode == "discrete" and not is_discrete:
            continue
        if mode == "continuous" and is_discrete:
            continue
            
        scenario_name = get_scenario_name(filename)
        
        candidates.append({
            "config": config_name,
            "scenario": scenario_name
        })
        
    # 2. Apply Whitelist (Smart Filter)
    if not whitelist:
        return candidates
        
    final_selection = []
    selected_indices = set()
    
    for term in whitelist:
        term_clean = term.replace(".py", "")
        
        # A. Check Exact Match
        exact_matches = []
        for i, cand in enumerate(candidates):
            if cand['config'] == term_clean:
                exact_matches.append(i)
                
        if exact_matches:
            # Found exact match(es) - usually 1
            for idx in exact_matches:
                selected_indices.add(idx)
        else:
            # B. Fallback to Substring Match
            sub_matches = []
            for i, cand in enumerate(candidates):
                if term_clean in cand['config']:
                    sub_matches.append(i)
                    
            for idx in sub_matches:
                selected_indices.add(idx)
                
    # Reconstruct list maintaining sort order
    experiments = [candidates[i] for i in sorted(list(selected_indices))]
        
    return experiments

def run_experiment(config_name, scenario_name, max_updates, args):
    print(f"\n{'='*60}")
    print(f" STARTING EXPERIMENT")
    print(f" Config  : {config_name}")
    print(f" Scenario: {scenario_name}")
    print(f"{'='*60}\n")
    
    cmd = [
        PYTHON_EXE, "main.py",
        "--config", config_name,
        "--scenario", scenario_name,
        "--max-updates", str(max_updates),
        "--eval-interval", str(EVAL_INTERVAL),
        "--eval-steps", str(args.eval_steps)
    ]
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        
        duration = time.time() - start_time
        print(f"\n>>> Verified: {config_name} DONE ({duration:.2f}s)")
        
    except subprocess.CalledProcessError as e:
        print(f"\n!!! FAILED: {config_name} (Exit Code: {e.returncode})")
    except Exception as e:
        print(f"\n!!! ERROR: {config_name} ({e})")

def run_numerical(config_name):
    """
    Runs numerical.py evaluation for the specific config.
    """
    print(f"\n{'='*60}")
    print(f" STARTING NUMERICAL EVALUATION")
    print(f" Config  : {config_name}")
    print(f" Policy  : DRL")
    print(f"{'='*60}\n")
    
    # Determine Mode
    is_discrete = config_name.startswith("D_")
    mode_arg = "discrete" if is_discrete else "continuous"
    
    cmd = [
        PYTHON_EXE, "numerical.py",
        "--mode", mode_arg,
        "--policy", "drl",
        "--select", config_name
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n>>> Verified: Numerical Eval for {config_name} DONE")
    except subprocess.CalledProcessError as e:
        print(f"\n!!! FAILED: Numerical Eval for {config_name} (Exit Code: {e.returncode})")
    except Exception as e:
        print(f"\n!!! ERROR: Numerical Eval for {config_name} ({e})")

def main():
    parser = argparse.ArgumentParser(description="Batch Experiment Runner")
    parser.add_argument("--mode", type=str, choices=["all", "discrete", "continuous"], default="all",
                        help="Filter configs: 'discrete' (D_*) or 'continuous' (Others) or 'all'")
    parser.add_argument("--max-updates", type=int, default=MAX_UPDATES,
                        help="Override max updates limit")
    parser.add_argument("--eval-steps", type=int, default=10000,
                        help="Evaluation scenario duration (default: 10000)")
    
    # [NEW] Whitelist Filter
    parser.add_argument("--filter", nargs='+', default=None,
                        help="List of specific config names (or substrings) to run")
    parser.add_argument("--select", nargs='+', default=None,
                        help="Alias for --filter (Select specific configs)")
    
    args = parser.parse_args()
    
    # Combine filter and select
    whitelist = []
    if args.filter: whitelist.extend(args.filter)
    if args.select: whitelist.extend(args.select)
    if not whitelist: whitelist = None
    
    experiments = get_experiment_list(args.mode, whitelist=whitelist)
    
    print(f"\nBatch Runner Started")
    print(f"Mode: {args.mode.upper()}")
    print(f"Configs Found: {len(experiments)}")
    print(f"Max Updates: {args.max_updates}")
    
    if not experiments:
        print("No experiments found matching criteria.")
        return

    print(f"\nList of experiments to run:")
    for e in experiments:
        print(f" - {e['config']}")

    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    for i, exp in enumerate(experiments):
        print(f"\nProcessing {i+1}/{len(experiments)}...")
        
        # 1. Run Training
        run_experiment(exp["config"], exp["scenario"], args.max_updates, args)
        
        # 2. Run Numerical Evaluation (Sequential)
        run_numerical(exp["config"])
        
        # Optional: Cool-down
        time.sleep(2)
        
    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()

#python main_batch.py --mode continuous --filter Homo_5R_8_ --eval-steps 20000