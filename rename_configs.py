
import os
import re
import json
import ast

def process_configs():
    config_dir = r"c:\Users\doyoon\Desktop\중견 4팀\BK_DRL\ver11\config"
    files = [f for f in os.listdir(config_dir) if f.endswith(".py") and f != "__init__.py"]
    
    # Group files by prefix
    # Pattern: Name_Parts_a_b.py where a and b are the last two parts
    # We want to keep everything up to the second to last underscore
    
    groups = {}
    
    for f in files:
        # Check if file matches pattern ~_~_a_b.py
        # Split by underscore
        parts = f.replace(".py", "").split("_")
        
        # We assume the last two are a and b.
        # Minimal parts length needs to be 4 to have a prefix and a, b (e.g. Type_R_LT_A_B)
        if len(parts) >= 4:
            # Reconstruct prefix
            # Example: Homo_1R_0_1_2 -> parts=['Homo', '1R', '0', '1', '2']
            # prefix parts: ['Homo', '1R', '0']
            prefix_parts = parts[:-2]
            prefix = "_".join(prefix_parts)
            
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(f)
        else:
            print(f"Skipping {f}, doesn't match expected pattern")

    for prefix, files in groups.items():
        # Sort files to find the best source (descending order generally prefers higher revision numbers like 4_4 over 1_2)
        files.sort(reverse=True)
        source_file_name = files[0]
        source_path = os.path.join(config_dir, source_file_name)
        target_path = os.path.join(config_dir, f"{prefix}.py")
        
        print(f"Processing group {prefix}: Source={source_file_name} -> Target={prefix}.py")
        
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Update network_arch
        # We'll use regex to find the network_arch block and replace it
        # Because we need valid python syntax, regex is safer than ast for just replacement if structure is consistent
        # But AST is better for correctness. Given the snippet provided, let's try to just replace the dict content if possible.
        
        # New network arch content
        new_arch = """
    "network_arch": {

        "embedding_dim_A": 32,
        "embedding_dim_C": 32,

        "mlp_inv_layers": [128, 64, 32],
        "mlp_ret_layers": [64, 32, 32],
        "policy_layers": [128, 128],
        "value_layers": [128, 128, 64],
    },"""

        # Regex to match network_arch key and its value (dict)
        # We assume matched braces.
        # A simple approach: find "network_arch": { and then find the closing brace that balances it?
        # Or since the structure is consistent in these config files, maybe just regex is enough.
        
        pattern = r'"network_arch"\s*:\s*\{[^{}]+\},'
        # The inner content might have nested braces (e.g. lists are [], but no nested dicts in the example).
        # The example values has lists [].
        # So [^{}]+ might not work if there are nested dicts. But here only lists.
        # Wait, the lists uses [].
        
        # Better pattern:
        # "network_arch":\s*\{.*?\}, (dotall)
        # Need to be careful about not consuming too much.
        # The closing brace for network_arch should be followed by a comma (mostly) or end of dict.
        
        # Let's try reading lines and identifying the block.
        lines = content.split('\n')
        new_lines = []
        in_network_arch = False
        skip = False
        
        for line in lines:
            if '"network_arch": {' in line:
                # Insert the new block
                # We need to preserve indentation if possible, but the template has its own indentation
                # The user provided snippet has 4 spaces indent.
                new_lines.append(new_arch.strip('\n'))  # Remove leading newline from my variable
                in_network_arch = True
                skip = True
            
            if in_network_arch:
                # Check for end of block.
                # Assuming standard formatting, the block ends with },
                if line.strip() == '},':
                    in_network_arch = False
                    skip = False
                    continue # The }, is already in new_arch
                elif line.strip() == '}':
                    # Last element might not have comma
                    in_network_arch = False
                    skip = False
                    continue
            
            if not skip:
                new_lines.append(line)
                
        new_content = '\n'.join(new_lines)
        
        # Verification
        if '"network_arch":' not in new_content:
             print(f"Error: Could not find network_arch in {source_file_name}")
             continue

        # Write to target
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        # Delete original files
        for f in files:
            path = os.path.join(config_dir, f)
            # If target path is same as source path (unexpected but possible if logic failed), don't delete
            if os.path.abspath(path) != os.path.abspath(target_path):
                os.remove(path)
                print(f"Deleted {f}")
            else:
                # If we overwrote the file in place (e.g. if filename didn't change), that's fine.
                # But here we are renaming.
                pass

if __name__ == "__main__":
    process_configs()
