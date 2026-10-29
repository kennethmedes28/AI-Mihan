"""LangSmith tracing for the flood agents, with personal details removed before anything is sent.

Two parts:

    enable_langsmith()    turns tracing on from the settings in ``backend/.env``
    traced_agent(name)    decorator that records one agent call as a LangSmith run

``.env`` settings (names only):

    LANGSMITH_KEY       the LangSmith API key
    LANGSMITH_TRACE     true / false: trace or not (``LANGSMITH_TRACING`` also works)
    LANGSMITH_PROJECT   the project the runs appear in (default "flood-gentic")

The LangSmith library itself reads ``LANGSMITH_API_KEY`` and ``LANGSMITH_TRACING``, so
``enable_langsmith`` copies the values over under those names. Tracing is off unless it is
asked for and a key is present, and an agent works the same whether it is on or off.

What is sent: the user's question and the agent's answer with phone-number-like digits replaced,
and only the city and precision of the user's location (never coordinates). Inputs named like
a secret (key, token, password) are never sent. Services, database connections and models are
replaced by their type name.
"""

import dataclasses
import os
import re

import langsmith
from dotenv import load_dotenv
from langsmith import Client, traceable

DEFAULT_PROJECT = "flood-gentic"
TRUE_VALUES = {"1", "true", "yes", "on"}

# Inputs with these words in their name are never sent (API keys, passwords, tokens).
SECRET_NAME_PATTERN = re.compile(r"key|token|secret|password|authorization", re.IGNORECASE)

# Seven or more digits, allowing spaces, dashes, dots, brackets and a leading plus: phone numbers.
PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-().]{5,}\d")
MAX_TEXT_CHARS = 4000  # runs stay small; the rest of a long text is cut
COORDINATE_NAMES = {"lat", "lng", "latitude", "longitude"}
COORDINATE_DECIMALS = 2  # about 1 km: enough to see the area, not the street


def enable_langsmith() -> bool:
    """Turn tracing on from ``.env``. Returns True when it is on."""
    load_dotenv()
    wanted = os.getenv("LANGSMITH_TRACE", os.getenv("LANGSMITH_TRACING", "")).strip().lower() in TRUE_VALUES
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGSMITH_KEY")
    if not (wanted and api_key):
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)

    # One shared client that cleans EVERY run before it is sent, including the runs LangGraph and
    # LangChain create by themselves (the whole conversation, each node and model call).
    langsmith.configure(client=Client(api_key=api_key, hide_inputs=sanitize, hide_outputs=sanitize))
    return True


def redact(text: str) -> str:
    """Replace phone-number-like digit groups and cut very long text."""
    return PHONE_PATTERN.sub("[number]", text)[:MAX_TEXT_CHARS]


def sanitize(value, name: str = ""):
    """Copy of any run input or output (dicts, lists, text) with private details removed.

    Phone-number-like digits are replaced, secrets are hidden, and coordinates are rounded to about 1 km.
    """
    if SECRET_NAME_PATTERN.search(name):
        return "[hidden]"
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, float) and name.lower() in COORDINATE_NAMES:
        return round(value, COORDINATE_DECIMALS)
    if isinstance(value, dict):
        return {key: sanitize(item, str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, name) for item in value]
    # LangChain messages are model objects and our own types are dataclasses: clean them as dictionaries.
    if hasattr(value, "model_dump"):
        return sanitize(value.model_dump(), name)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sanitize(dataclasses.asdict(value), name)
    return value


def clean_value(value):
    """A version of ``value`` that is safe and small enough to send."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    location = getattr(value, "location", None)  # a ChatContext: keep only the coarse place
    if hasattr(value, "get_connection"):
        place = {"city": location.city, "precision": location.precision} if location else None
        return {"location": place}
    if getattr(value, "text", None) is not None:  # an AgentReply
        return {"text": redact(value.text), "pending_intent": getattr(value, "pending_intent", None)}
    return f"<{type(value).__name__}>"


def clean_inputs(inputs: dict) -> dict:
    """Clean every input; anything named like a secret is replaced, whatever its value."""
    return {
        name: "[hidden]" if SECRET_NAME_PATTERN.search(name) else clean_value(value)
        for name, value in inputs.items()
    }


def clean_outputs(outputs) -> dict:
    cleaned = clean_value(outputs)
    return cleaned if isinstance(cleaned, dict) else {"output": cleaned}


def traced_agent(name: str, run_type: str = "chain"):
    """Decorator: record each call of an agent function as a LangSmith run called ``name``.

    Inside the chat graph the run is nested under the graph's own run. With tracing off it does nothing.
    """
    return traceable(
        name=name,
        run_type=run_type,
        process_inputs=clean_inputs,
        process_outputs=clean_outputs,
    )


enable_langsmith()  # runs when the first agent is imported, before any call is traced
