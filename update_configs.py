import os
import glob
import re

def update_configs():
    config_dir = "config"
    files = glob.glob(os.path.join(config_dir, "*.py"))
    
    # Target parameter
    param_key = "mono_coef"
    param_value = 0.5
    
    count_updated = 0
    count_skipped = 0
    
    for f_path in files:
        if "__init__.py" in f_path: continue
        
        with open(f_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if already exists
        if f'"{param_key}"' in content or f"'{param_key}'" in content:
            # Optionally update value? Or skip?
            # Let's update if exists, or append if not.
            # Using simple regex to replace value if exists
            
            # Pattern: "mono_coef": <number>,
            pattern = r"(['\"]" + param_key + r"['\"]\s*:\s*)([\d\.]+)(,?)"
            if re.search(pattern, content):
                # Replace value
                new_content = re.sub(pattern, f"'{param_key}': {param_value},", content)
                if new_content != content:
                    with open(f_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"[Updated] {os.path.basename(f_path)}: Set {param_key} = {param_value}")
                    count_updated += 1
                else:
                    print(f"[Skipped] {os.path.basename(f_path)}: Value already correct.")
                    count_skipped += 1
                continue
                
        # If not exists, insert it
        # Insert before the closing brace of config dictionary
        # Find the last occurrence of '}' that closes 'config = {'
        
        # Heuristic: Insert after 'config = {' line
        if "config = {" in content:
            lines = content.splitlines()
            new_lines = []
            inserted = False
            for line in lines:
                new_lines.append(line)
                if "config = {" in line and not inserted:
                    # Add strictly indented line
                    new_lines.append(f"    '{param_key}': {param_value},")
                    inserted = True
            
            if inserted:
                with open(f_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(new_lines))
                print(f"[Inserted] {os.path.basename(f_path)}: Added {param_key} = {param_value}")
                count_updated += 1
            else:
                print(f"[Error] Could not find insertion point in {f_path}")
        else:
            print(f"[Error] 'config = {{' not found in {f_path}")

    print(f"\nSummary: Updated {count_updated}, Skipped {count_skipped}, Total {len(files)}")

if __name__ == "__main__":
    update_configs()
