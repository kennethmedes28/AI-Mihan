"""Test settings: never send traces to LangSmith from the tests (fake data would fill the project and use its monthly allowance)."""

import os

# Set before any agent is imported; utils/tracing.py then leaves tracing off.
os.environ["LANGSMITH_TRACE"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
