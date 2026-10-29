"""Copy the CSV datasets and a saved FAISS run into Supabase.

Run from backend/ so the project imports and .env are found:

    uv run python ../scripts/load_to_supabase.py              # datasets + latest FAISS run
    uv run python ../scripts/load_to_supabase.py --dry-run    # check the files, change nothing

Tables created (all with row-level security on and no policies, so only the
backend's direct Postgres connection can read or write them):

    ncr_locations              datasets/ncr_locations.csv
    ncr_weather_daily          datasets/ncr_weather_daily_2020-2026.csv
    ncr_flood_features_daily   datasets/ncr_flood_features_daily.csv
    chunks                     the newest folder in faiss_vectors/ (vectors read back from index.faiss)

Re-running is safe: dataset tables are emptied and reloaded, chunks are upserted.
"""

import argparse
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_DIR / "datasets"
FAISS_DIR = REPO_DIR / "faiss_vectors"

sys.path.insert(0, str(REPO_DIR / "backend"))  # so `src` and `checkpointer` can be imported
load_dotenv()

# The PDF was renamed after the saved FAISS run was built.
DOC_ID_RENAMES = {"world-bank": "MMFMP"}
SOURCE_RENAMES = {"world-bank.pdf": "MMFMP.pdf"}

DATASET_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ncr_locations (
    location_id  integer PRIMARY KEY,
    latitude     double precision NOT NULL,
    longitude    double precision NOT NULL,
    elevation_m  double precision,
    coord_group  integer,
    city         text NOT NULL
);

CREATE TABLE IF NOT EXISTS ncr_weather_daily (
    location_id                        integer NOT NULL REFERENCES ncr_locations (location_id),
    date                               date    NOT NULL,
    precipitation_sum_mm               double precision,
    temperature_2m_max_c               double precision,
    temperature_2m_min_c               double precision,
    shortwave_radiation_sum_mjm2       double precision,
    wind_speed_10m_max_kmh             double precision,
    wind_gusts_10m_max_kmh             double precision,
    temperature_2m_max_from_hourly_c   double precision,
    temperature_2m_min_from_hourly_c   double precision,
    river_discharge_m3s                double precision,
    PRIMARY KEY (location_id, date)
);

CREATE TABLE IF NOT EXISTS ncr_flood_features_daily (
    location_id               integer NOT NULL REFERENCES ncr_locations (location_id),
    date                      date    NOT NULL,
    precipitation_sum_mm      double precision,
    wind_speed_10m_max_kmh    double precision,
    wind_gusts_10m_max_kmh    double precision,
    river_discharge_m3s       double precision,
    rain_max_1h_mm            double precision,
    rain_max_3h_mm            double precision,
    rain_max_6h_mm            double precision,
    elevation_m               double precision,
    rain_3d_mm                double precision,
    rain_7d_mm                double precision,
    rain_14d_mm               double precision,
    wet_streak_days           integer,
    discharge_change_1d_pct   double precision,
    discharge_max_3d_m3s      double precision,
    discharge_pctile          double precision,
    PRIMARY KEY (location_id, date)
);

ALTER TABLE ncr_locations            ENABLE ROW LEVEL SECURITY;
ALTER TABLE ncr_weather_daily        ENABLE ROW LEVEL SECURITY;
ALTER TABLE ncr_flood_features_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks                   ENABLE ROW LEVEL SECURITY;
"""

# (table, csv file name). Locations come first because the other two point at it.
DATASET_TABLES = [
    ("ncr_locations", "ncr_locations.csv"),
    ("ncr_weather_daily", "ncr_weather_daily_2020-2026.csv"),
    ("ncr_flood_features_daily", "ncr_flood_features_daily.csv"),
]


def read_csv_header(csv_path: Path) -> list[str]:
    """Return the column names of a CSV (utf-8-sig drops the byte-order mark some files start with)."""
    with open(csv_path, encoding="utf-8-sig", newline="") as file:
        return next(csv.reader(file))


def load_dataset(conn, table: str, csv_path: Path) -> int:
    """Empty ``table`` and fill it from the CSV with COPY. Empty cells become NULL."""
    columns = ", ".join(read_csv_header(csv_path))
    conn.execute(f"TRUNCATE {table} CASCADE")
    copy_sql = f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, HEADER true)"
    with conn.cursor() as cursor, cursor.copy(copy_sql) as copy:
        with open(csv_path, encoding="utf-8-sig") as file:
            while block := file.read(1 << 20):
                copy.write(block)
    return conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]


def latest_faiss_run() -> Path:
    """The newest run folder in faiss_vectors/ (folder names start with a timestamp)."""
    runs = sorted(path for path in FAISS_DIR.iterdir() if path.is_dir())
    if not runs:
        raise SystemExit(f"No FAISS run found in {FAISS_DIR}")
    return runs[-1]


def rebuild_chunks(run_dir: Path):
    """Read a saved run and return ``(chunks, embeddings)`` with the renamed PDF applied."""
    from src.rag.faiss_store import load_run
    from src.rag.models import Chunk

    index, records, _ = load_run(run_dir)
    if index.ntotal != len(records):
        raise SystemExit(f"Index has {index.ntotal} vectors but chunks.json has {len(records)} chunks")

    # Row i of the index belongs to records[i]; IndexFlatIP can hand the vectors back.
    embeddings = index.reconstruct_n(0, index.ntotal).tolist()

    chunks = []
    for record in records:
        metadata = dict(record["metadata"])
        if "source" in metadata:
            metadata["source"] = SOURCE_RENAMES.get(metadata["source"], metadata["source"])
        doc_id = DOC_ID_RENAMES.get(record["doc_id"], record["doc_id"])
        chunks.append(Chunk(doc_id, record["chunk_idx"], record["content"], metadata))
    return chunks, embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="read and check the files, write nothing")
    args = parser.parse_args()

    run_dir = latest_faiss_run()
    chunks, embeddings = rebuild_chunks(run_dir)
    print(f"FAISS run {run_dir.name}: {len(chunks)} chunks, vector size {len(embeddings[0])}")
    for table, file_name in DATASET_TABLES:
        print(f"{table}: {len(read_csv_header(DATASETS_DIR / file_name))} columns from {file_name}")
    if args.dry_run:
        return

    from src.rag import vector_store
    from src.services.fetch_postgre_database import get_connection

    with get_connection() as conn:
        vector_store.ensure_schema(conn)
        conn.execute(DATASET_SCHEMA_SQL)
        for table, file_name in DATASET_TABLES:
            print(f"{table}: loaded {load_dataset(conn, table, DATASETS_DIR / file_name)} rows")

        hashes = [vector_store.content_hash(chunk.content) for chunk in chunks]
        vector_store.upsert_chunks(conn, chunks, embeddings, hashes)
        total = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        print(f"chunks: {total} rows in table")


if __name__ == "__main__":
    main()
