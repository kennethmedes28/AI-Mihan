"""Database settings for the LangGraph checkpointer.

Reads the Supabase Postgres connection details from environment variables
(see ``.env.example``) so no credentials live in the code.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()


@dataclass(frozen=True)
class CheckpointerDBConfig:
    """Connection settings used to build the checkpointer's connection pool."""

    # Postgres connection string, e.g. the Supabase "connection pooling" URL.
    connection_url: str
    # Pool size limits. Keep max_size small: Supabase caps total connections.
    min_pool_size: int = 1
    max_pool_size: int = 5

    @property
    def connection_kwargs(self) -> dict:
        """Extra arguments passed to every connection in the pool.

        - autocommit: the checkpointer runs its own setup and writes, and
          expects each statement to be committed immediately.
        - prepare_threshold=None: Supabase's transaction pooler (port 6543)
          does not support prepared statements, so turn them off.
        - dict_row: the checkpointer reads rows by column name.
        """
        return {
            "autocommit": True,
            "prepare_threshold": None,
            "row_factory": dict_row,
        }


def load_db_config() -> CheckpointerDBConfig:
    """Build the config from environment variables.

    Required:
        SUPABASE_DB_URL: Postgres connection string.
    Optional:
        CHECKPOINTER_POOL_MIN / CHECKPOINTER_POOL_MAX: pool size limits.
    """
    connection_url = os.getenv("SUPABASE_DB_URL")
    if not connection_url:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set. Add it to backend/.env "
            "(see .env.example)."
        )

    return CheckpointerDBConfig(
        connection_url=connection_url,
        min_pool_size=int(os.getenv("CHECKPOINTER_POOL_MIN", "1")),
        max_pool_size=int(os.getenv("CHECKPOINTER_POOL_MAX", "5")),
    )
