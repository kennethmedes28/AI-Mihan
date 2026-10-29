"""Add the new rows of the dataset CSVs to Supabase. Append only.

Run from backend/ (or anywhere, with the Supabase settings in the environment):

    uv run python ../scripts/sync_datasets_to_supabase.py

For each table the CSV rows are copied into a temporary staging table, then
inserted with ``ON CONFLICT DO NOTHING``. A row whose (location_id, date) is
already in Supabase is left exactly as it is, and nothing is ever updated or
deleted, so running it twice in a row changes nothing.

Tables (created if missing) and their CSV files:

    ncr_locations              datasets/ncr_locations.csv
    ncr_weather_daily          datasets/ncr_weather_daily_2020-2026.csv
    ncr_flood_features_daily   datasets/ncr_flood_features_daily.csv

Connection settings: see backend/.env.example (SUPABASE_DB_URL, or project ref + password + pooler host).
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_DIR / "datasets"

sys.path.insert(0, str(REPO_DIR / "backend"))  # so `checkpointer` and `src` can be imported
sys.path.insert(0, str(REPO_DIR / "scripts"))  # so the table definitions can be shared
load_dotenv()

from load_to_supabase import DATASET_SCHEMA_SQL, DATASET_TABLES, read_csv_header  # noqa: E402

# The columns that identify a row in each table.
KEY_COLUMNS = {
    "ncr_locations": "location_id",
    "ncr_weather_daily": "location_id, date",
    "ncr_flood_features_daily": "location_id, date",
}


def append_new_rows(conn, table: str, csv_path: Path) -> int:
    """Insert the CSV rows that are not in ``table`` yet. Returns how many were added."""
    columns = ", ".join(read_csv_header(csv_path))

    # Everything below runs in one transaction, so the staging table lives on one connection.
    with conn.transaction():
        conn.execute(f"CREATE TEMP TABLE staging (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP")
        copy_sql = f"COPY staging ({columns}) FROM STDIN WITH (FORMAT csv, HEADER true)"
        with conn.cursor() as cursor, cursor.copy(copy_sql) as copy:
            with open(csv_path, encoding="utf-8-sig") as file:
                while block := file.read(1 << 20):
                    copy.write(block)

        result = conn.execute(
            f"INSERT INTO {table} ({columns}) SELECT {columns} FROM staging "
            f"ON CONFLICT ({KEY_COLUMNS[table]}) DO NOTHING"
        )
        return result.rowcount


def main() -> None:
    from src.services.fetch_postgre_database import get_connection

    with get_connection() as conn:
        conn.execute(DATASET_SCHEMA_SQL)  # CREATE TABLE IF NOT EXISTS, safe to repeat
        for table, file_name in DATASET_TABLES:
            added = append_new_rows(conn, table, DATASETS_DIR / file_name)
            print(f"{table}: {added} new rows added")


if __name__ == "__main__":
    main()
