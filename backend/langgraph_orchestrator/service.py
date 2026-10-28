"""Starts the chat graph with its real services (database, saved conversations).

Use it once at server start-up; ``graph.py`` itself knows nothing about how
the database or the checkpointer are opened.

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with open_chat_graph(llm, retriever) as graph:
            app.state.graph = graph
            yield

    # per request
    reply = await ask(app.state.graph, question, thread_id=conversation_id, location=gps_fix)
"""

import logging
import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator

import psycopg
from langchain_core.language_models import BaseChatModel

from checkpointer.postgres_checkpointer import open_postgres_checkpointer
from langgraph_orchestrator.graph import build_graph
from src.rag.retriever import HybridRetriever
from src.services.fetch_postgre_database import get_connection

logger = logging.getLogger(__name__)


class SharedConnection:
    """One database connection for the agents, opened on first use and reopened if it dropped.

    The agents only read evacuation centers and write reports, so one
    connection is enough and cannot leak the way a new one per question would.
    Calling the object returns the open connection.
    """

    def __init__(self, open_connection=get_connection):
        self._open_connection = open_connection
        self._connection: psycopg.Connection | None = None
        self._lock = threading.Lock()

    def __call__(self) -> psycopg.Connection:
        with self._lock:
            if self._connection is None or self._connection.closed or self._connection.broken:
                self._connection = self._open_connection()
            return self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None and not self._connection.closed:
                self._connection.close()
            self._connection = None


@asynccontextmanager
async def open_chat_graph(
    llm: BaseChatModel,
    retriever: HybridRetriever | None = None,
    ors_api_key: str | None = None,
) -> AsyncIterator:
    """Open the checkpointer and the database connection, and yield the compiled graph.

    Both are closed when the ``async with`` block ends.
    """
    shared_connection = SharedConnection()
    try:
        async with open_postgres_checkpointer() as checkpointer:
            yield build_graph(
                llm,
                retriever,
                checkpointer,
                get_connection=shared_connection,
                ors_api_key=ors_api_key,
            )
    finally:
        shared_connection.close()
