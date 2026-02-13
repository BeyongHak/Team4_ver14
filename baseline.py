
import os
import csv
import ast
import numpy as np

def generate_baseline_table(folder_name):
    # 1. Define Paths
    base_dir = os.getcwd()
    source_path = os.path.join(base_dir, "result_SAA", folder_name, "SAA_policy_map_updated.csv")
    
    # Output Directory
    output_dir = os.path.join(base_dir, "baseline")
    os.makedirs(output_dir, exist_ok=True)
    
    target_path = os.path.join(output_dir, f"{folder_name}.csv")

    if not os.path.exists(source_path):
        print(f"[Skip] Source file not found: {source_path}")
        return

    print(f"Processing {folder_name}...")

    # 2. Load SAA Map
    # We load into a dictionary that we will modify in place.
    saa_map = {}
    
    try:
        with open(source_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader) # Skip header
            
            for row in reader:
                if not row: continue
                try:
                    # Key format: "[0, 0, ...]" -> tuple
                    state_list = ast.literal_eval(row[0])
                    state_tuple = tuple(state_list)
                    val = float(row[1])
                    saa_map[state_tuple] = val
                except:
                    continue
    except Exception as e:
        print(f"Error reading {source_path}: {e}")
        return

    if not saa_map:
        print("Empty SAA map.")
        return

    # 3. Enforce Diagonal Monotonicity
    # Condition: Action(a+1, b+1...) >= Action(a, b...)
    # Logic: If Action(Next) < Action(Prev), set Action(Next) = Action(Prev)
    
    # To do this correctly, we must process states in an order that guarantees 
    # the 'previous' diagonal predecessor has already been finalized.
    # Sorting by 'sum of elements' guarantees this, because sum(prev) < sum(current).
    
    sorted_states = sorted(saa_map.keys(), key=lambda s: sum(s))
    
    modified_count = 0
    final_saa_map = {}
    
    # We iterate and build a new map (or update in place, but new map is cleaner logic)
    # Actually, we can update saa_map in place if we iterate in sorted order.
    
    for state in sorted_states:
        # Determine Predecessor: (k1-1, k2-1, ...)
        # Check if all elements > 0. If any is 0, it has no diagonal predecessor in the positive orthant.
        
        has_predecessor = all(k > 0 for k in state)
        current_val = saa_map[state]
        
        if has_predecessor:
            prev_state = tuple(k - 1 for k in state)
            
            # The predecessor must exist in the map if the map is complete (filled).
            # If not found (e.g. boundary issues), we skip correction.
            if prev_state in saa_map:
                prev_val = saa_map[prev_state]
                
                # Check Constraint
                if current_val < prev_val:
                    # Violates Monotonicity -> Fix
                    # We ALWAYS fix strictly (even for small diffs) to ensure monotonicity
                    saa_map[state] = prev_val
                    
                    # Count only significant violations (>= 1.0) as per user request
                    if (prev_val - current_val) >= 1.0:
                        modified_count += 1
        
    print(f"  - Monotonicity Correction: Fixed {modified_count} violations.")

    # 4. Save Final Map & Meta
    # Global Max (Post-Correction)
    global_saa_max = max(saa_map.values())
    
    # Save CSV
    try:
        # Resort keys primarily by state for readability
        output_keys = sorted(saa_map.keys())
        
        with open(target_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["State", "Baseline_Value"])
            
            for k in output_keys:
                k_list = list(k)
                val = saa_map[k]
                writer.writerow([str(k_list), f"{val:.6f}"])
        print(f"  - Saved {len(output_keys)} baseline entries to {target_path}")
        
        # Save Meta
        meta_path = os.path.join(output_dir, f"{folder_name}_meta.txt")
        with open(meta_path, 'w') as f:
            f.write(f"{global_saa_max}\n")
            f.write(f"Violations: {modified_count}\n")
        print(f"  - Saved Meta (Max={global_saa_max}, Violations={modified_count}) to {meta_path}")
        
    except Exception as e:
        print(f"Error saving {target_path} or meta: {e}")

def main():
    root_dir = os.path.join(os.getcwd(), "result_SAA")
    if not os.path.exists(root_dir):
        print("result_SAA directory not found.")
        return

    # List all subdirectories in result_SAA
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    if not subdirs:
        print("No subdirectories found in result_SAA.")
        return
        
    for d in subdirs:
        generate_baseline_table(d)

if __name__ == "__main__":
    main()
