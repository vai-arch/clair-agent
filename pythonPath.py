from pathlib import Path
import os

# 1️⃣ Set workspace root (adjust if this script is inside workspace root)
workspace = Path(__file__).parent.resolve()

# 2️⃣ Find all 'src' folders under week-*/day-*/src
src_folders = [str(p.resolve()) for p in workspace.glob("week-*/day-*/src") if p.is_dir()]

if not src_folders:
    print("No src folders found under week-*/day-*/src")
else:
    # 3️⃣ Determine path separator based on OS
    path_sep = ";" if os.name == "nt" else ":"

    # 4️⃣ Create .env content
    env_content = f"PYTHONPATH={path_sep.join(src_folders)}\n"

    # 5️⃣ Write to .env in workspace root
    env_file = workspace / ".env"
    env_file.write_text(env_content)

    print(f".env file created with {len(src_folders)} src folders at {env_file}")
