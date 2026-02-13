import os
import csv
import shutil
import sys

def sort_summary(target_path):
    if not os.path.exists(target_path):
        print(f"[Warning] {target_path} does not exist.")
        return

    print(f"Sorting {target_path}...")
    
    # 1. Create a backup for safety
    backup_path = target_path + ".bak"
    try:
        shutil.copy2(target_path, backup_path)
        print(f"  [Backup] Created backup at {backup_path}")
    except Exception as e:
        print(f"  [Error] Failed to create backup: {e}")
        return  # Stop if backup fails

    try:
        rows = []
        # Use utf-8-sig for Excel compatibility and to handle BOM
        with open(target_path, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                print("  [Error] File is empty.")
                return
            
            # Identify Config column
            try:
                cfg_idx = header.index("Config")
            except ValueError:
                # Fallback: Check if 0-th is likely config
                # Usually Config is first
                cfg_idx = 0
            
            for row in reader:
                if row:
                    rows.append(row)
        
        # Sort rows by Config column (Alphabetical)
        rows.sort(key=lambda x: x[cfg_idx])
        
        # Write back with utf-8-sig
        with open(target_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
            
        print(f"  [Done] Sorted {len(rows)} entries.")
        
    except Exception as e:
        print(f"  [Error] Failed to sort {target_path}: {e}")
        print(f"  [Info] You can restore from {backup_path}")

def main():
    # Targets
    targets = [
        os.path.join("result_C", "summary.csv"),
        os.path.join("result_D", "summary.csv"),
        os.path.join("result_SAA", "summary.csv")
    ]
    
    for t in targets:
        sort_summary(t)

if __name__ == "__main__":
    main()
