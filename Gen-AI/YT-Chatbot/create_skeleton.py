import os

project_name = "chat_with_youtube"  # folder name
structure = {
    "": ["README.md", "requirements.txt", ".env.example", ".gitignore", "Dockerfile", "Makefile"],
    "app": ["__init__.py", "main.py"],
    "app/pages": ["1_Summary.py", "2_Glossary.py", "3_UsageStats.py"],
    "app/components": ["video_display.py", "chat_box.py", "progress_bar.py"],
    "core": ["__init__.py", "config.py", "logging.py", "utils.py"],
    "data": ["cache", "transcripts", "tmp"],
    "ingest": ["__init__.py", "fetch_transcript.py", "preprocess.py", "chunking.py", "embed_store.py", "pipeline.py"],
    "retrieval": ["__init__.py", "retriever.py", "chain.py", "prompts.py", "memory.py"],
    "services": ["__init__.py", "gemini.py", "fallback.py", "quota.py"],
    "tests": ["__init__.py", "test_ingest.py", "test_retrieval.py", "test_chain.py", "test_prompts.py"],
    "scripts": ["run_ingest.py", "run_summary.py", "run_chat.py"],
}

def make_dirs_and_files(base, struct):
    for folder, files in struct.items():
        dirpath = os.path.join(base, folder)
        os.makedirs(dirpath, exist_ok=True)
        for fname in files:
            path = os.path.join(dirpath, fname)
            # If it's a directory (ends with no extension and name same as file in data), create it
            if fname in ["cache", "transcripts", "tmp"] and (folder == "data"):
                os.makedirs(path, exist_ok=True)
            else:
                # touch file
                if not os.path.exists(path):
                    with open(path, "w") as f:
                        # optionally put a placeholder comment
                        f.write(f"# {fname}\n")
    print(f"Project skeleton created at {base}")

if __name__ == "__main__":
    make_dirs_and_files(".", structure)
