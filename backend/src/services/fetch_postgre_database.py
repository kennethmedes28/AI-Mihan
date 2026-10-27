"""Read app data (evacuation centers, community reports) from Supabase Postgres.

Tables follow docs/backend_architecture.md section 6. Query functions take an
open connection from ``get_connection()``; rows come back as dictionaries.
"""

import psycopg

from checkpointer.db_config import load_db_config

MAX_LIMIT = 50  # no single query returns more than this many rows


def get_connection() -> psycopg.Connection:
    """Open a connection using SUPABASE_DB_URL (same settings as the checkpointer)."""
    config = load_db_config()
    return psycopg.connect(config.connection_url, **config.connection_kwargs)


def _build_query(base_sql: str, conditions: list[str], order_by: str, limit: int) -> str:
    """Add WHERE, ORDER BY and LIMIT to a query. Only fixed text is inserted, never user input."""
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    return f"{base_sql}{where} ORDER BY {order_by} LIMIT {min(int(limit), MAX_LIMIT)}"


def fetch_evacuation_centers(
    conn, city: str | None = None, barangay: str | None = None, limit: int = 20
) -> list[dict]:
    """Return evacuation centers, optionally limited to a city and / or barangay."""
    conditions, values = [], []
    if city:
        conditions.append("city ILIKE %s")
        values.append(city)
    if barangay:
        conditions.append("barangay ILIKE %s")
        values.append(barangay)

    sql = _build_query(
        "SELECT name, city, barangay, capacity, lat, lng FROM evacuation_centers",
        conditions,
        order_by="city, barangay, name",
        limit=limit,
    )
    return conn.execute(sql, values).fetchall()


def fetch_recent_reports(
    conn, city: str | None = None, hazard: str | None = None, limit: int = 10
) -> list[dict]:
    """Return the newest community reports, optionally limited to a city and / or hazard.

    Reports are community-submitted and unverified (see ``trust_level``).
    """
    conditions, values = [], []
    if city:
        conditions.append("city ILIKE %s")
        values.append(city)
    if hazard:
        conditions.append("hazard ILIKE %s")
        values.append(hazard)

    sql = _build_query(
        "SELECT hazard, city, barangay, body, status, trust_level, created_at FROM reports",
        conditions,
        order_by="created_at DESC",
        limit=limit,
    )
    return conn.execute(sql, values).fetchall()
