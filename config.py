"""
Shared configuration loader for Day 22 Lab.
This file loads environment variables and provides a helper for configuring LLMs/Embeddings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the root .env file
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# --- Configuration Values ---
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "day22-langsmith-lab"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or ""
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2") or os.getenv("LANGSMITH_TRACING") or "true"

# Strip potential outer quotes if the user defined them with quotes in .env
if LANGCHAIN_PROJECT.startswith('"') and LANGCHAIN_PROJECT.endswith('"'):
    LANGCHAIN_PROJECT = LANGCHAIN_PROJECT[1:-1]
if LANGCHAIN_API_KEY.startswith('"') and LANGCHAIN_API_KEY.endswith('"'):
    LANGCHAIN_API_KEY = LANGCHAIN_API_KEY[1:-1]

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Dynamically select endpoint and key based on model type
if "gpt" in DEFAULT_MODEL.lower() or "text-embedding-3" in EMBEDDING_MODEL.lower():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = "https://api.openai.com/v1"
else:
    OPENAI_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    OPENAI_API_BASE = os.getenv("DASHSCOPE_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

# Set LangSmith system environment variables for tracing
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Also set OpenAI compatible keys for default usage
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE



def verify_config():
    """Verify that configuration can be loaded correctly."""
    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {LANGCHAIN_PROJECT}")
    print(f"   OpenAI endpoint   : {OPENAI_API_BASE}")
    print(f"   Default LLM model : {DEFAULT_MODEL}")
    print(f"   Embedding model   : {EMBEDDING_MODEL}")


if __name__ == "__main__":
    verify_config()
