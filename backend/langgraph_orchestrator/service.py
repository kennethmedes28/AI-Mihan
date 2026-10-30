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
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langchain_core.language_models import BaseChatModel

from checkpointer.postgres_checkpointer import open_postgres_checkpointer
from langgraph_orchestrator.graph import build_graph
from src.rag.retriever import HybridRetriever
from src.services.shared_connection import SharedConnection

logger = logging.getLogger(__name__)


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
