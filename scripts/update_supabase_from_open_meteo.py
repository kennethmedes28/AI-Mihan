"""Top up the Supabase dataset tables straight from Open-Meteo. No CSV files and no local checkout needed.

Run from anywhere with the Supabase settings in the environment (see backend/.env.example):

    uv run python ../scripts/update_supabase_from_open_meteo.py --dry-run   # read Supabase, show the plan, change nothing
    uv run python ../scripts/update_supabase_from_open_meteo.py             # fetch the missing days and write them
    uv run python ../scripts/update_supabase_from_open_meteo.py --end-date 2026-11-30
    uv run python ../scripts/update_supabase_from_open_meteo.py --refresh-days 7   # also re-fetch the last 7 days

Steps:

    1. Read from Supabase: the 17 sample points (ncr_locations) and every row of ncr_weather_daily.
    2. Find the first day any location is missing, and also go back --refresh-days days (default
       REFRESH_DAYS, 2), because Open-Meteo revises its newest days. A day Open-Meteo returns blank is
       skipped, never stored as blank, so it is filled in on a later run.
    3. Fetch only those days from Open-Meteo (the same code as update_open_meteo_datasets.py).
    4. Work out the flood-feature columns. They need earlier days (rolling rain, river-flow rank), which
       come from the rows already in Supabase.
    5. Write both tables in ONE transaction: missing days are inserted, and the refreshed days replace the
       earlier values. Rows older than the window are never touched, and nothing is ever deleted.
       If any step fails, nothing is written.

The CSV files in datasets/ are not read or changed. Use update_open_meteo_datasets.py when you want them too.
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "backend"))  # so `src` and `checkpointer` can be imported
sys.path.insert(0, str(REPO_DIR / "scripts"))  # so the Open-Meteo code can be shared
load_dotenv(REPO_DIR / "backend" / ".env")  # a missing file is fine: in GitHub Actions the settings are secrets

from update_open_meteo_datasets import (  # noqa: E402
    CHUNK_DAYS,
    MAX_CALLS_PER_RUN,
    REFRESH_DAYS,
    SECONDS_BETWEEN_CALLS,
    TIMEZONE,
    WEATHER_ROUNDING,
    OpenMeteoClient,
    RateLimitError,
    build_new_features,
    fetch_point,
    find_first_missing_day,
    split_into_windows,
)

WEATHER_TABLE = "ncr_weather_daily"
FEATURES_TABLE = "ncr_flood_features_daily"
KEY_COLUMNS = ["location_id", "date"]

MISSING_TABLES_MESSAGE = (
    "The dataset tables are missing or empty in Supabase. Create and fill them once with "
    "scripts/load_to_supabase.py, then use this script to keep them current."
)


def table_columns(conn, table: str) -> list[str]:
    """Column names of a table, in table order (the order the DataFrames and COPY use)."""
    return [column.name for column in conn.execute(f"SELECT * FROM {table} LIMIT 0").description]


def read_supabase_state(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(locations, weather)`` as they are in Supabase. Dates are ISO text, like in the CSV files."""
    locations = pd.DataFrame(conn.execute("SELECT * FROM ncr_locations ORDER BY location_id").fetchall())
    weather = pd.DataFrame(
        conn.execute(f"SELECT * FROM {WEATHER_TABLE} ORDER BY location_id, date").fetchall(),
        columns=table_columns(conn, WEATHER_TABLE),
    )
    if locations.empty or weather.empty:
        raise SystemExit(MISSING_TABLES_MESSAGE)
    weather["date"] = weather["date"].astype(str)
    return locations, weather


def plan_window(weather: pd.DataFrame, locations: pd.DataFrame, end: date, refresh_days: int = REFRESH_DAYS) -> date:
    """The first day to fetch: the first missing day of any location, or the start of the refresh window, whichever is earlier."""
    return min(find_first_missing_day(weather, locations), end - timedelta(days=refresh_days - 1))


def fetch_new_rows(
    weather: pd.DataFrame, locations: pd.DataFrame, feature_columns: list[str], start: date, end: date, max_calls: int
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fetch ``start..end`` from Open-Meteo and return ``(weather_rows, feature_rows)``, or None if there is nothing new.

    ``weather`` is what Supabase already holds. Its rows from ``start`` on are being replaced, so they are left out
    of the history the rolling features are calculated from.
    """
    weather_columns = list(weather.columns)
    history = weather[weather["date"] < start.isoformat()]

    windows = split_into_windows(start, end, CHUNK_DAYS)
    points = locations.drop_duplicates(["latitude", "longitude"])[["latitude", "longitude"]]
    planned_calls = len(points) * 2 * len(windows)  # weather + river discharge for every window
    if planned_calls > max_calls:
        raise SystemExit(f"{planned_calls} calls needed but the limit is {max_calls}. "
                         "Raise --max-calls or use an earlier --end-date.")

    client = OpenMeteoClient(max_calls, SECONDS_BETWEEN_CALLS)
    days_by_coordinate = {}
    for point in points.itertuples():
        print(f"fetching {point.latitude}, {point.longitude}")
        days_by_coordinate[(point.latitude, point.longitude)] = fetch_point(
            client, point.latitude, point.longitude, windows, windows
        )

    # Every location gets the days of its coordinate.
    pieces = []
    for location in locations.itertuples():
        days = days_by_coordinate[(location.latitude, location.longitude)].copy()
        days.insert(0, "location_id", location.location_id)
        pieces.append(days)
    new_days = pd.concat(pieces, ignore_index=True).sort_values(KEY_COLUMNS)
    if new_days.empty:
        return None

    new_weather = new_days[weather_columns].round(WEATHER_ROUNDING)
    hourly_columns = new_days[["location_id", "date", "rain_max_1h_mm", "rain_max_3h_mm", "rain_max_6h_mm"]].round(1)
    new_features = build_new_features(
        pd.concat([history, new_weather], ignore_index=True), new_weather, hourly_columns, locations, feature_columns
    )
    return new_weather, new_features


def upsert_rows(conn, table: str, rows: pd.DataFrame) -> None:
    """Insert the rows; a row whose (location_id, date) already exists gets the new values.

    The rows are copied into a temporary table first (fast, and empty cells become NULL), then inserted.
    """
    columns = list(rows.columns)
    column_list = ", ".join(columns)
    updates = ", ".join(f"{name} = EXCLUDED.{name}" for name in columns if name not in KEY_COLUMNS)
    with conn.transaction():  # a savepoint when the caller already opened a transaction
        conn.execute(f"CREATE TEMP TABLE staging (LIKE {table} INCLUDING DEFAULTS)")
        with conn.cursor() as cursor, cursor.copy(
            f"COPY staging ({column_list}) FROM STDIN WITH (FORMAT csv, HEADER true)"
        ) as copy:
            copy.write(rows.to_csv(index=False))
        conn.execute(
            f"INSERT INTO {table} ({column_list}) SELECT {column_list} FROM staging "
            f"ON CONFLICT (location_id, date) DO UPDATE SET {updates}"
        )
        conn.execute("DROP TABLE staging")  # not ON COMMIT DROP: both tables are written in one outer transaction


def write_rows(conn, new_weather: pd.DataFrame, new_features: pd.DataFrame) -> None:
    """Write both tables or neither."""
    with conn.transaction():
        upsert_rows(conn, WEATHER_TABLE, new_weather)
        upsert_rows(conn, FEATURES_TABLE, new_features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--end-date", help="last day to fetch, YYYY-MM-DD (default: today in Manila)")
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS_PER_RUN, help=f"stop if more calls are needed (default {MAX_CALLS_PER_RUN})")
    parser.add_argument("--refresh-days", type=int, default=REFRESH_DAYS,
                        help=f"also re-fetch this many days up to the end date and replace their values (default {REFRESH_DAYS})")
    parser.add_argument("--dry-run", action="store_true", help="read Supabase and show the plan; fetch and write nothing")
    args = parser.parse_args()

    from src.services.fetch_postgre_database import get_connection

    if args.refresh_days < 1:
        raise SystemExit("--refresh-days must be at least 1.")
    end = date.fromisoformat(args.end_date) if args.end_date else datetime.now(ZoneInfo(TIMEZONE)).date()

    with get_connection() as conn:
        locations, weather = read_supabase_state(conn)
        feature_columns = table_columns(conn, FEATURES_TABLE)
        last_days = weather.groupby("location_id")["date"].max()
        start = plan_window(weather, locations, end, args.refresh_days)
        print(f"Supabase: {len(weather)} weather rows, {last_days.min()} is the oldest 'last day' of any location.")
        print(f"Fetching {start} to {end} from Open-Meteo.")
        if start > end:
            print("Already up to date.")
            return
        if args.dry_run:
            points = len(locations.drop_duplicates(["latitude", "longitude"]))
            print(f"Dry run: up to {points * 2 * len(split_into_windows(start, end, CHUNK_DAYS))} calls. Nothing fetched or written.")
            return

        try:
            fetched = fetch_new_rows(weather, locations, feature_columns, start, end, args.max_calls)
        except RateLimitError as error:
            raise SystemExit(f"{error}\nNothing was written to Supabase.")
        if fetched is None:
            print("Open-Meteo has no new complete days yet. Nothing written.")
            return

        new_weather, new_features = fetched
        already_there = len(weather[weather["date"] >= start.isoformat()])
        write_rows(conn, new_weather, new_features)
        print(f"Wrote {len(new_weather)} rows to each table ({new_weather['date'].min()} to {new_weather['date'].max()}); "
              f"{already_there} of them replaced earlier values.")


if __name__ == "__main__":
    main()
