"""The main chat model for the agents: Muse Wonder at high reasoning effort.

Muse's API is OpenAI-compatible, so the standard LangChain OpenAI client is used
with Meta's base URL. The weather and document agents take the model as an
argument, so this is the one place that decides which model they get:

    llm = build_main_llm()
    async with open_chat_graph(llm, retriever) as graph:
        ...

Settings can be overridden with environment variables (useful for trying
another model without editing code):

    MUSE_API               the API key (required)
    MUSE_CHAT_MODEL        model name, default ``muse-wonder-1.0``
    MUSE_REASONING_EFFORT  ``minimal``, ``low``, ``medium`` or ``high`` (default)

High effort gives the most careful answers but the slowest ones: much of the
time is hidden thinking, billed as output (see docs/agent_benchmarks.md, where
``minimal`` was 2 to 4 times faster on the Spark models).
"""

import os

from langchain_openai import ChatOpenAI

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-wonder-1.0"
DEFAULT_REASONING_EFFORT = "high"
REQUEST_TIMEOUT_SECONDS = 120.0  # high effort can think for a while before the first word


def build_main_llm(api_key: str | None = None) -> ChatOpenAI:
    """Build the chat model used by the weather and document agents."""
    api_key = api_key or os.getenv("MUSE_API")
    if not api_key:
        raise RuntimeError("MUSE_API is not set")

    return ChatOpenAI(
        model=os.getenv("MUSE_CHAT_MODEL", DEFAULT_MODEL),
        base_url=BASE_URL,
        api_key=api_key,
        reasoning_effort=os.getenv("MUSE_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
        timeout=REQUEST_TIMEOUT_SECONDS,
        # No max_tokens: the model's hidden thinking counts against it, and a low cap
        # can leave nothing for the answer (a 300-token test returned an empty reply).
    )
