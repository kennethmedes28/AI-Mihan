"""Postgres-backed checkpointer for the LangGraph chat graph.

The checkpointer saves the graph state after every step, keyed by
``thread_id``, so a conversation can continue across requests and server
restarts. State is stored in the same Supabase Postgres database as the
rest of the app.

Typical use with FastAPI (see langgraph_orchestrator/service.py, which does this; open once at startup, close at shutdown):

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with open_postgres_checkpointer() as checkpointer:
            app.state.graph = build_graph(llm, retriever, checkpointer)
            yield

    # Per request, use one thread id per conversation:
    config = {"configurable": {"thread_id": conversation_id}}
    await app.state.graph.ainvoke({"messages": [...]}, config)
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from checkpointer.db_config import CheckpointerDBConfig, load_db_config


@asynccontextmanager
async def open_postgres_checkpointer(
    config: CheckpointerDBConfig | None = None,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Open a connection pool and yield a ready-to-use checkpointer.

    Creates the checkpoint tables on first run, and closes the pool when the
    ``async with`` block exits.
    """
    config = config or load_db_config()

    # open=False + explicit open() avoids the deprecated auto-open in the
    # pool constructor.
    pool = AsyncConnectionPool(
        conninfo=config.connection_url,
        min_size=config.min_pool_size,
        max_size=config.max_pool_size,
        kwargs=config.connection_kwargs,
        open=False,
    )
    await pool.open()

    try:
        checkpointer = AsyncPostgresSaver(pool)
        # Safe to call every start: only creates missing tables/migrations.
        await checkpointer.setup()
        yield checkpointer
    finally:
        await pool.close()
