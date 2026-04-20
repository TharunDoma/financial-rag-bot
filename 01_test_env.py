"""
01_test_env.py
--------------
Purpose : Verify the project environment is configured correctly.
          Loads the Gemini API key from .env and confirms it is present.
          Run this BEFORE writing any RAG or pipeline code.

DE Concept: This is the "environment validation" step — in production
            pipelines (e.g., Airflow, dbt), every DAG or job starts by
            asserting that required credentials and configs exist BEFORE
            any data is touched. Fail fast, fail early.

Usage:
    python 01_test_env.py
"""

import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Step 1: Load variables from .env into the process environment.
#         python-dotenv reads the key=value pairs and injects them so that
#         os.environ can access them — the API key never lives in source code.
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Step 2: Read the key from the environment (NOT from a hardcoded string).
# ---------------------------------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# Step 3: Validate — fail loudly if the key is missing or still a placeholder.
# ---------------------------------------------------------------------------
if not api_key:
    raise EnvironmentError(
        "\n[ERROR] GEMINI_API_KEY is not set.\n"
        "  → Open your .env file and replace 'your_gemini_api_key_here' "
        "with your actual key."
    )

if api_key == "your_gemini_api_key_here":
    raise EnvironmentError(
        "\n[ERROR] GEMINI_API_KEY is still the placeholder value.\n"
        "  → Open your .env file and paste in your real Gemini API key."
    )

# ---------------------------------------------------------------------------
# Step 4: Print a masked confirmation so the key is never echoed to the terminal.
#         In enterprise pipelines this is called "secret masking" — the same
#         pattern used by GitHub Actions, CircleCI, and AWS Secrets Manager.
# ---------------------------------------------------------------------------
masked = api_key[:6] + "*" * (len(api_key) - 6)
print("\n✅ Environment check passed!")
print(f"   GEMINI_API_KEY loaded: {masked}")
print("\n🚀 Your foundation is solid. Ready to build the RAG pipeline.\n")
