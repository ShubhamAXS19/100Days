import os
import shutil

def safe_move(src, dst):
    if not os.path.exists(src):
        print(f"⚠️  Skipping {src} (not found)")
        return
    if os.path.exists(dst):
        print(f"✅  {dst} already exists, skipping move")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(f"📦  Moved {src} → {dst}")

def safe_create(path, content=""):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        print(f"🆕  Created {path}")
    else:
        print(f"✅  {path} already exists, skipping create")

def refactor_project():
    # Backend base
    backend_dirs = [
        "backend/app/agents",
        "backend/app/tools",
        "backend/app/utils",
        "backend/app/data",
        "backend/app/tests",
    ]
    for d in backend_dirs:
        os.makedirs(d, exist_ok=True)

    # Move files if exist
    safe_move("app", "backend/app")
    safe_move("requirements.txt", "backend/requirements.txt")
    safe_move("Dockerfile", "backend/Dockerfile")
    safe_move(".env.example", "backend/.env.example")

    # Rename server.py → main.py
    old_server = "backend/app/server.py"
    new_main = "backend/app/main.py"
    if os.path.exists(old_server) and not os.path.exists(new_main):
        shutil.move(old_server, new_main)
        print("📝  Renamed server.py → main.py")

    # Create agent placeholders
    agents = {
        "document_agent.py": "# Handles document parsing and OCR\n",
        "extraction_agent.py": "# Extracts trade entities and checks sanctions\n",
        "decision_agent.py": "# Generates compliance decision and report\n",
    }
    for fname, content in agents.items():
        safe_create(f"backend/app/agents/{fname}", content)

    # Create frontend structure
    os.makedirs("frontend/components", exist_ok=True)
    safe_create("frontend/streamlit_app.py", "# Streamlit entrypoint\n")
    safe_create("frontend/requirements.txt", "streamlit\nrequests\n")
    safe_create("frontend/Dockerfile", "# Dockerfile for Streamlit frontend\n")
    safe_create("frontend/.env.example", "# .env.example\n")

    for fname, content in {
        "upload_box.py": "# Handles file uploads\n",
        "progress_tracker.py": "# Displays progress\n",
        "decision_view.py": "# Shows compliance decision\n",
    }.items():
        safe_create(f"frontend/components/{fname}", content)

    print("\n🎉 Project skeleton ready!")

if __name__ == "__main__":
    refactor_project()
