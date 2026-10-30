"""Starts the real services the API needs: the database, the chat model, document search and the chat graph.

Each optional piece fails softly, because a person in a flood still needs the parts that work:

    * no chat model key      -> weather and document answers fail, but emergency, evacuation,
                                route, contacts and reports still work (they use no model)
    * document search fails  -> the document agent gives the fixed safe answer
    * the database is down   -> the API starts anyway and /health says so

Only a wrong setting (for example a missing checkpointer package) stops the start-up, with a message that says what to do.
"""

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver

from langgraph_orchestrator.graph import build_graph
from src.services.shared_connection import SharedConnection
from utils.llm import build_main_llm

logger = logging.getLogger(__name__)


class UnavailableChatModel(BaseChatModel):
    """Stands in for the chat model when no key is set. Using it raises, and the graph then gives its error reply."""

    @property
    def _llm_type(self) -> str:
        return "unavailable"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("The chat model is not configured (MUSE_API is not set).")


def build_llm() -> tuple[BaseChatModel, bool]:
    """Return ``(model, configured)``. Without a key, a stand-in that raises when used."""
    try:
        return build_main_llm(), True
    except RuntimeError as error:
        logger.warning("%s Weather and document answers are off.", error)
        return UnavailableChatModel(), False


def build_retriever(connection: SharedConnection):
    """Build the document search from the chunks in the database, or return None if that fails.

    The keyword index is built once here, so chunks ingested later need a restart to be found.
    """
    try:
        from src.rag.embeddings import embed_query
        from src.rag.retriever import HybridRetriever

        return HybridRetriever.from_database(connection(), embed_query)
    except Exception as error:  # Database down, packages missing, empty table, and so on.
        logger.warning("Document search is off: %s", error)
        return None


async def open_checkpointer(stack: AsyncExitStack, choice: str):
    """Open the conversation store: Postgres (saved) or memory (lost on restart)."""
    if choice == "memory":
        logger.warning("CHECKPOINTER=memory: conversations are lost when the server restarts.")
        return MemorySaver()

    try:
        from checkpointer.postgres_checkpointer import open_postgres_checkpointer
    except ImportError as error:
        raise RuntimeError(
            "The Postgres checkpointer needs langgraph-checkpoint-postgres and psycopg-pool. "
            "Run 'uv sync', or set CHECKPOINTER=memory for local work."
        ) from error
    return await stack.enter_async_context(open_postgres_checkpointer())


@asynccontextmanager
async def real_services(app: FastAPI):
    """Lifespan of the real app: open everything, hand it to the routers through ``app.state``, close it at shutdown."""
    settings = app.state.settings

    async with AsyncExitStack() as stack:
        connection = SharedConnection()
        stack.callback(connection.close)

        llm, app.state.llm_ready = build_llm()
        retriever = build_retriever(connection) if settings.enable_rag else None
        app.state.retriever_ready = retriever is not None

        checkpointer = await open_checkpointer(stack, settings.checkpointer)
        app.state.graph = build_graph(
            llm,
            retriever,
            checkpointer,
            get_connection=connection,
            ors_api_key=settings.ors_api_key,
        )
        app.state.get_connection = connection
        yield
