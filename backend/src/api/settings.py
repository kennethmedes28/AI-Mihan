"""Settings for the API, read from environment variables (and ``backend/.env``).

Nothing secret is stored here. The database and model keys are read where they
are used (``checkpointer/db_config.py`` and ``utils/llm.py``).

Variables, all optional:

    CORS_ORIGINS         comma-separated browser origins allowed to call the API
                         (default ``http://localhost:3000``)
    CHECKPOINTER         ``postgres`` (default, conversations survive restarts) or
                         ``memory`` (for local work; conversations are lost on restart)
    ENABLE_RAG           ``true`` (default) loads the document search at start-up
    TRUST_PROXY_HEADERS  ``true`` only behind a proxy you control (Railway, nginx). The
                         client address is then read from ``X-Forwarded-For``; anyone can
                         fake that header when there is no proxy, so the default is ``false``
    CHAT_PER_MINUTE      calls to POST /chat allowed per client per minute (default 30)
    REPORTS_PER_HOUR     calls to POST /reports allowed per client per hour (default 10)
    REQUESTS_PER_HOUR    calls to POST /requests allowed per client per hour (default 10)
    ORS_API_KEY          OpenRouteService key for the route agent
    STAFF_API_KEY        secret for the /staff endpoints (list and review reports and requests).
                         At least 24 characters, e.g. from `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
                         Without it the /staff endpoints do not exist at all
    STAFF_PER_MINUTE     calls to /staff allowed per client per minute (default 60); this also slows key guessing
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

CHECKPOINTER_CHOICES = ("postgres", "memory")
MIN_STAFF_KEY_CHARS = 24  # a short key can be guessed


@dataclass(frozen=True)
class Settings:
    """Everything the API can be configured with."""

    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    checkpointer: str = "postgres"
    enable_rag: bool = True
    trust_proxy_headers: bool = False
    chat_per_minute: int = 30
    reports_per_hour: int = 10
    requests_per_hour: int = 10
    ors_api_key: str | None = None
    staff_api_key: str | None = None
    staff_per_minute: int = 60


def read_bool(name: str, default: bool) -> bool:
    """Read a true/false variable. Accepts true, 1, yes, on (any case)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


def load_settings() -> Settings:
    """Build the settings from the environment, checking the values that can be wrong."""
    load_dotenv()

    checkpointer = os.getenv("CHECKPOINTER", "postgres").strip().lower()
    if checkpointer not in CHECKPOINTER_CHOICES:
        raise RuntimeError(f"CHECKPOINTER must be one of {CHECKPOINTER_CHOICES}, not {checkpointer!r}")

    staff_api_key = os.getenv("STAFF_API_KEY") or None
    if staff_api_key and len(staff_api_key) < MIN_STAFF_KEY_CHARS:
        raise RuntimeError(f"STAFF_API_KEY must be at least {MIN_STAFF_KEY_CHARS} characters long.")

    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return Settings(
        cors_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
        checkpointer=checkpointer,
        enable_rag=read_bool("ENABLE_RAG", True),
        trust_proxy_headers=read_bool("TRUST_PROXY_HEADERS", False),
        chat_per_minute=int(os.getenv("CHAT_PER_MINUTE", "30")),
        reports_per_hour=int(os.getenv("REPORTS_PER_HOUR", "10")),
        requests_per_hour=int(os.getenv("REQUESTS_PER_HOUR", "10")),
        ors_api_key=os.getenv("ORS_API_KEY") or None,
        staff_api_key=staff_api_key,
        staff_per_minute=int(os.getenv("STAFF_PER_MINUTE", "60")),
    )
