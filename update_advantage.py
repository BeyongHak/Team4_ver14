
import os
import glob
import re

config_dir = r"c:\Users\doyoon\Desktop\중견 4팀\BK_DRL\ver13\config"
py_files = glob.glob(os.path.join(config_dir, "*.py"))

for file_path in py_files:
    if "__init__.py" in file_path: continue
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace 'normalize_advantage': True with False (Handling potential spacing)
    new_content = re.sub(r'\"normalize_advantage\"\s*:\s*True', '"normalize_advantage": False', content)
    
    if content != new_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {os.path.basename(file_path)}")
    else:
        print(f"Skipped {os.path.basename(file_path)} (No change needed)")
