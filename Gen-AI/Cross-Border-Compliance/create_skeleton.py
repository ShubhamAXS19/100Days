import os

project_name = "trade_compliance_checker"

structure = {
    "": ["README.md", "requirements.txt", ".env.example", "Dockerfile", "Makefile"],
    "app": ["__init__.py", "server.py"],
    "app/agents": ["__init__.py", "document_agent.py", "extraction_agent.py", "decision_agent.py"],
    "app/tools": ["__init__.py", "ocr_tool.py", "pdf_parser_tool.py", "sanctions_db_tool.py", "embedding_tool.py"],
    "data": ["sample_docs", "sanctions_list.csv"],
    "tests": ["__init__.py", "test_agents.py", "test_tools.py"],
    "scripts": ["run_demo.py"],
}

placeholder_content = {
    "server.py": "# server.py\n# LangServe entrypoint placeholder\n",
    "document_agent.py": "# document_agent.py\nclass DocumentAgent:\n    def process(self, file_path):\n        return {'text': 'dummy text'}\n",
    "extraction_agent.py": "# extraction_agent.py\nclass ExtractionAgent:\n    def extract(self, text):\n        return {'entities': []}\n",
    "decision_agent.py": "# decision_agent.py\nclass DecisionAgent:\n    def decide(self, entities):\n        return {'decision': 'pending'}\n",
    "ocr_tool.py": "# ocr_tool.py\n# TODO: implement OCR logic\n",
    "pdf_parser_tool.py": "# pdf_parser_tool.py\n# TODO: implement PDF parsing logic\n",
    "sanctions_db_tool.py": "# sanctions_db_tool.py\n# TODO: implement sanctions DB check\n",
    "embedding_tool.py": "# embedding_tool.py\n# TODO: implement embeddings / semantic search\n",
    "test_agents.py": "# test_agents.py\n# TODO: write unit tests for agents\n",
    "test_tools.py": "# test_tools.py\n# TODO: write unit tests for tools\n",
    "run_demo.py": "# run_demo.py\n# Quick demo script placeholder\n",
    "README.md": "# Trade Compliance Checker\n\nProject skeleton for multi-agent LangChain compliance app.\n",
    "requirements.txt": "# requirements.txt\nlangchain\nlangserve\nfastapi\nuvicorn\n",
    ".env.example": "# .env.example\n# Add your API keys here\n",
    "Dockerfile": "# Dockerfile\n# Placeholder Dockerfile\n",
    "Makefile": "# Makefile\n# Placeholder for build / run commands\n",
}

def make_dirs_and_files(base, struct, placeholders):
    for folder, files in struct.items():
        dirpath = os.path.join(base, folder)
        os.makedirs(dirpath, exist_ok=True)
        for fname in files:
            path = os.path.join(dirpath, fname)
            if fname in ["sample_docs"]:  # create folder instead of file
                os.makedirs(path, exist_ok=True)
            else:
                if not os.path.exists(path):
                    with open(path, "w") as f:
                        f.write(placeholders.get(fname, f"# {fname}\n"))
    print(f"Project skeleton created at {os.path.abspath(base)}")

if __name__ == "__main__":
    make_dirs_and_files(".", structure, placeholder_content)
