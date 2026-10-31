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
    """Return evacuation centers, optionally limited to a city and / or barangay.

    ``city`` is the key the agents use ("paranaque", "quezon city"). The table holds the
    names as written in the source ("Parañaque City", "Pateros"), so the match ignores
    accents and the trailing word "City": the key only has to be the start of the name.
    """
    conditions, values = [], []
    if city:
        conditions.append("translate(lower(city), 'ñ', 'n') ILIKE %s")
        values.append(city.lower().replace("ñ", "n") + "%")
    if barangay:
        conditions.append("barangay ILIKE %s")
        values.append(barangay)

    sql = _build_query(
        "SELECT name, city, barangay, type, latitude AS lat, longitude AS lng, notes FROM evacuation_centers",
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


def fetch_report_points(conn, hours: int = 6, hazard: str = "flood", limit: int = MAX_LIMIT) -> list[dict]:
    """Return recent reports that have a map position (needs sql/001_reports_geo.sql).

    Rejected reports are left out. Used to mark flooded spots for routing.
    """
    sql = _build_query(
        "SELECT city, barangay, body, lat, lng, status, trust_level, created_at FROM reports",
        [
            "hazard ILIKE %s",
            "lat IS NOT NULL",
            "lng IS NOT NULL",
            "status <> 'rejected'",
            "created_at >= now() - make_interval(hours => %s)",
        ],
        order_by="created_at DESC",
        limit=limit,
    )
    return conn.execute(sql, [hazard, int(hours)]).fetchall()


def insert_report(
    conn,
    hazard: str,
    city: str,
    barangay: str | None,
    body: str,
    lat: float | None = None,
    lng: float | None = None,
) -> None:
    """Store a community report. It starts unverified, with community trust."""
    conn.execute(
        "INSERT INTO reports (hazard, city, barangay, body, status, trust_level, lat, lng) "
        "VALUES (%s, %s, %s, %s, 'unverified', 'community', %s, %s)",
        [hazard, city, barangay, body, lat, lng],
    )


def insert_request(
    conn,
    kind: str,
    city: str,
    barangay: str | None,
    details: str,
    people_count: int | None = None,
    contact_phone: str | None = None,
    lat: float | None = None,
    lng: float | None = None,
) -> None:
    """Store a supply or assistance request. It starts unverified (needs sql/002_requests.sql)."""
    conn.execute(
        "INSERT INTO requests (kind, city, barangay, details, people_count, contact_phone, lat, lng, status) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'unverified')",
        [kind, city, barangay, details, people_count, contact_phone, lat, lng],
    )


# --- Staff review of reports and requests --------------------------------------------------------
# The table name is never taken from a request: it must be one of these two fixed words.
STAFF_TABLES = ("reports", "requests")

REPORT_STATUSES = ("unverified", "verified", "rejected")
REQUEST_STATUSES = ("unverified", "verified", "assigned", "closed")


def fetch_reports_for_staff(
    conn, status: str | None = None, city: str | None = None, hazard: str | None = None, limit: int = 20
) -> list[dict]:
    """Return the newest reports with everything, including the id and position (staff only)."""
    conditions, values = [], []
    for column, value in (("status", status), ("city", city), ("hazard", hazard)):
        if value:
            conditions.append(f"{column} ILIKE %s")  # the column name is fixed text, the value is a parameter
            values.append(value)
    sql = _build_query(
        "SELECT id, hazard, city, barangay, body, status, trust_level, lat, lng, created_at FROM reports",
        conditions,
        order_by="created_at DESC",
        limit=limit,
    )
    return conn.execute(sql, values).fetchall()


def fetch_requests_for_staff(
    conn, status: str | None = None, city: str | None = None, kind: str | None = None, limit: int = 20
) -> list[dict]:
    """Return the newest requests with everything, including the contact phone (staff only)."""
    conditions, values = [], []
    for column, value in (("status", status), ("city", city), ("kind", kind)):
        if value:
            conditions.append(f"{column} ILIKE %s")
            values.append(value)
    sql = _build_query(
        "SELECT id, kind, city, barangay, details, people_count, contact_phone, lat, lng, status, created_at FROM requests",
        conditions,
        order_by="created_at DESC",
        limit=limit,
    )
    return conn.execute(sql, values).fetchall()


def update_status(conn, table: str, row_id, status: str) -> bool:
    """Set the status of one report or request. Returns False when there is no row with that id."""
    if table not in STAFF_TABLES:
        raise ValueError(f"unknown table: {table}")
    cursor = conn.execute(f"UPDATE {table} SET status = %s WHERE id = %s", [status, row_id])
    return cursor.rowcount > 0


def summarize_table(conn, table: str) -> dict:
    """Counts for monitoring: total, by status, by city, and when the oldest unverified row arrived."""
    if table not in STAFF_TABLES:
        raise ValueError(f"unknown table: {table}")
    by_status = conn.execute(f"SELECT status, count(*) AS n FROM {table} GROUP BY status").fetchall()
    by_city = conn.execute(f"SELECT city, count(*) AS n FROM {table} GROUP BY city ORDER BY n DESC").fetchall()
    oldest = conn.execute(f"SELECT min(created_at) AS oldest FROM {table} WHERE status = 'unverified'").fetchone()
    return {
        "total": sum(row["n"] for row in by_status),
        "by_status": {row["status"]: row["n"] for row in by_status},
        "by_city": {row["city"]: row["n"] for row in by_city},
        "oldest_unverified_at": oldest["oldest"] if oldest else None,
    }
