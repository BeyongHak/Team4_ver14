
import os
import csv
import ast
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def check_monotonicity(mode, config_name, target_file, output_dir):
    """
    Check Monotonicity for a given policy map file.
    Returns summary dict.
    """
    
    # 1. Load Data
    data_map = {}
    
    if not os.path.exists(target_file):
        print(f"[Skip] File not found: {target_file}")
        return {
            "Config": config_name,
            "Status": "Skipped (File Not Found)",
            "Violations": 0,
            "Total_Checks": 0,
            "Max_Violation": 0.0
        }

    print(f"[{mode.upper()}] Loading map: {target_file}")
    
    try:
        df = pd.read_csv(target_file)
        
        state_col = df.columns[0]
        val_col = df.columns[1] 
        
        def parse_state(s):
            try:
                return tuple(ast.literal_eval(s))
            except:
                return None
                
        df['State_Tuple'] = df[state_col].apply(parse_state)
        df = df.dropna(subset=['State_Tuple'])
        
        data_map = dict(zip(df['State_Tuple'], df[val_col]))
        
    except Exception as e:
        print(f"[Error] Failed to read {target_file}: {e}")
        return {
            "Config": config_name,
            "Status": f"Error: {e}",
            "Violations": 0,
            "Total_Checks": 0,
            "Max_Violation": 0.0
        }

    if not data_map:
        return {
            "Config": config_name,
            "Status": "Empty Map",
            "Violations": 0,
            "Total_Checks": 0,
            "Max_Violation": 0.0
        }

    # 2. Check Monotonicity
    violations = []
    total_checks = 0
    max_violation = 0.0
    
    sorted_keys = sorted(data_map.keys())
    
    for current_state in tqdm(sorted_keys, desc=f"Checking {config_name}"):
        current_val = data_map[current_state]
        
        if all(x > 0 for x in current_state):
            prev_state = tuple(x - 1 for x in current_state)
            
            if prev_state in data_map:
                prev_val = data_map[prev_state]
                total_checks += 1
                
                if current_val < prev_val:
                    diff = prev_val - current_val
                    if diff > 1e-6:
                        violations.append({
                            "Current_State": current_state,
                            "Prev_State": prev_state,
                            "Current_Val": current_val,
                            "Prev_Val": prev_val,
                            "Violation": diff
                        })
                        if diff > max_violation:
                            max_violation = diff

    # 3. Report & Save
    if violations:
        print(f"  [Fail] Found {len(violations)} violations / {total_checks} checks.")
        
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "monotonicity_violations.csv")
        
        v_df = pd.DataFrame(violations)
        v_df.to_csv(out_path, index=False)
        print(f"  Saved detailed violations to {out_path}")
        
        return {
            "Config": config_name,
            "Status": "Fail",
            "Violations": len(violations),
            "Total_Checks": total_checks,
            "Max_Violation": max_violation
        }
    else:
        print(f"  [Pass] No violations found.")
        return {
            "Config": config_name,
            "Status": "Pass",
            "Violations": 0,
            "Total_Checks": total_checks,
            "Max_Violation": 0.0
        }


def main():
    parser = argparse.ArgumentParser(description="Monotonicity Checker")
    parser.add_argument("--mode", type=str, required=True, choices=["saa", "baseline", "drl"],
                        help="Mode: 'saa' (result_SAA), 'baseline' (baseline/), 'drl' (result_C/D)")
    parser.add_argument("--filter", nargs='+', default=None, help="Filter specific configs")
    
    args = parser.parse_args()
    
    root_dir = os.getcwd()
    tasks = [] 
    output_root_base = ""
    
    if args.mode == "saa":
        source_root = os.path.join(root_dir, "result_SAA")
        output_root_base = os.path.join(root_dir, "check_SAA")
        pattern = "SAA_policy_map_updated.csv"
        
        if os.path.exists(source_root):
            for d in os.listdir(source_root):
                if os.path.isdir(os.path.join(source_root, d)):
                    tasks.append({
                        "config": d,
                        "file": os.path.join(source_root, d, pattern),
                        "out": os.path.join(output_root_base, d)
                    })
                    
    elif args.mode == "baseline":
        source_root = os.path.join(root_dir, "baseline")
        output_root_base = os.path.join(root_dir, "check_baseline")
        
        if os.path.exists(source_root):
            files = [f for f in os.listdir(source_root) if f.endswith(".csv")]
            for f in files:
                config_name = f.replace(".csv", "")
                tasks.append({
                    "config": config_name,
                    "file": os.path.join(source_root, f),
                    "out": os.path.join(output_root_base, config_name)
                })

    elif args.mode == "drl":
        source_root = os.path.join(root_dir, "result_C") 
        output_root_base = os.path.join(root_dir, "check_drl")
        pattern = "DRL_policy_map.csv"
        
        if os.path.exists(source_root):
            for d in os.listdir(source_root):
                if os.path.isdir(os.path.join(source_root, d)):
                    tasks.append({
                        "config": d,
                        "file": os.path.join(source_root, d, pattern),
                        "out": os.path.join(output_root_base, d)
                    })

    # Execute Tasks & Collect Summary
    summary_results = []
    
    count = 0
    if not os.path.exists(output_root_base):
        os.makedirs(output_root_base, exist_ok=True)

    for task in tasks:
        if args.filter:
            if not any(f in task["config"] for f in args.filter):
                continue
                
        res = check_monotonicity(args.mode, task["config"], task["file"], task["out"])
        if res:
            summary_results.append(res)
        count += 1
        
    if count == 0:
        print("No matching files found.")
    else:
        # Save Summary CSV
        summary_path = os.path.join(output_root_base, f"summary_{args.mode}.csv")
        summary_df = pd.DataFrame(summary_results)
        
        # Reorder columns
        cols = ["Config", "Status", "Violations", "Total_Checks", "Max_Violation"]
        summary_df = summary_df[cols]
        
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[Summary] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
