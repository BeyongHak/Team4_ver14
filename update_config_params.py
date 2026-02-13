
import os
import glob
import re

config_dir = r"c:\Users\doyoon\Desktop\중견 4팀\BK_DRL\ver13\config"
py_files = glob.glob(os.path.join(config_dir, "*.py"))

for file_path in py_files:
    if "__init__.py" in file_path: continue
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 1. Update entropy_coef start to 0.005
    # Look for "start": 0.05 or "start": 5e-2 etc. Assuming standard formatting based on previous view.
    # We will use a regex that targets the specific structure to be safe.
    
    # Regex for entropy start:  "start": <number>,
    # We want to replace 0.05 with 0.005
    # Warning: simpler replace might be safer if we are sure of the string "start": 0.05
    
    new_content = re.sub(r'("start":\s*)0\.05', r'\1 0.005', content)
    
    # 2. Update normalize_advantage to True
    new_content = re.sub(r'("normalize_advantage"\s*:\s*)False', r'\1True', new_content)
    
    if content != new_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {os.path.basename(file_path)}")
    else:
        print(f"Skipped {os.path.basename(file_path)} (No change needed)")
