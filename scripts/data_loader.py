"""Load the last 7 days of Supabase data for the Streamlit frontend.

The Streamlit app (``frontend_streamlit/``) calls this module to show recent
rain, river discharge and flood features. It reads the same Supabase tables
that the weather agent uses (see ``docs/dataset_updates.md`` section 7):

    ncr_locations              one point per NCR city (17 cities)
    ncr_weather_daily          daily weather + river discharge
    ncr_flood_features_daily   daily flood features (rolling rain, wet streak)

All functions return a ``pandas.DataFrame`` sorted oldest first, so the app
can pass the result straight to ``st.line_chart`` / ``st.dataframe``. Results
are cached for 30 minutes because the tables only change once a day.

Run from ``backend/`` so the installed packages are found, for example::

    uv run python ../scripts/data_loader.py   # prints the 7-day overview
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_DIR / "backend"))  # so `src` and `checkpointer` can be imported
load_dotenv()

DEFAULT_DAYS = 7
MAX_DAYS = 30
CACHE_SECONDS = 30 * 60  # the tables only change once a day

# One row per day for a city. A city can have several sample points, so rain
# is averaged and discharge / wet streak take the highest point (same rule as
# the weather agent's history query). The city name is compared without the
# accent ("Las Piñas" = "las pinas") and without a trailing note
# ("Pateros (municipality)" = "pateros").
FLOOD_FEATURES_SQL = """
SELECT f.date,
       l.city,
       ROUND(AVG(f.precipitation_sum_mm)::numeric, 1)::float AS rain_mm,
       ROUND(AVG(f.rain_max_1h_mm)::numeric, 1)::float       AS rain_max_1h_mm,
       ROUND(AVG(f.rain_max_3h_mm)::numeric, 1)::float       AS rain_max_3h_mm,
       ROUND(AVG(f.rain_max_6h_mm)::numeric, 1)::float       AS rain_max_6h_mm,
       ROUND(AVG(f.rain_3d_mm)::numeric, 1)::float           AS rain_3d_mm,
       ROUND(AVG(f.rain_7d_mm)::numeric, 1)::float           AS rain_7d_mm,
       ROUND(AVG(f.rain_14d_mm)::numeric, 1)::float          AS rain_14d_mm,
       MAX(f.wet_streak_days)                                AS wet_streak_days,
       MAX(f.river_discharge_m3s)::float                     AS river_discharge_m3s
FROM ncr_flood_features_daily f
JOIN ncr_locations l USING (location_id)
WHERE f.date > (SELECT max(date) FROM ncr_flood_features_daily) - %s
GROUP BY f.date, l.city
ORDER BY l.city, f.date
"""

WEATHER_SQL = """
SELECT w.date,
       l.city,
       ROUND(AVG(w.precipitation_sum_mm)::numeric, 1)::float AS rain_mm,
       ROUND(AVG(w.temperature_2m_max_c)::numeric, 1)::float AS temp_max_c,
       ROUND(AVG(w.temperature_2m_min_c)::numeric, 1)::float AS temp_min_c,
       ROUND(AVG(w.wind_speed_10m_max_kmh)::numeric, 1)::float AS wind_max_kmh,
       ROUND(AVG(w.wind_gusts_10m_max_kmh)::numeric, 1)::float AS wind_gust_kmh,
       MAX(w.river_discharge_m3s)::float                     AS river_discharge_m3s
FROM ncr_weather_daily w
JOIN ncr_locations l USING (location_id)
WHERE w.date > (SELECT max(date) FROM ncr_weather_daily) - %s
GROUP BY w.date, l.city
ORDER BY l.city, w.date
"""

LOCATIONS_SQL = """
SELECT location_id, city, latitude, longitude, elevation_m
FROM ncr_locations
ORDER BY city
"""

CITY_FILTER = " AND translate(lower(regexp_replace(l.city, ' \\(.*\\)$', '')), 'ñ', 'n') = %s"


def _clamp_days(days: int) -> int:
    """Keep ``days`` inside 1..MAX_DAYS so a bad app input cannot scan years."""
    return max(1, min(int(days), MAX_DAYS))


def _query(sql: str, params: list) -> "object":
    """Run ``sql`` and return the rows as a DataFrame (empty on no rows)."""
    import pandas as pd
    from src.services.fetch_postgre_database import get_connection

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            columns = [d.name for d in cursor.description]
            rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def _cached(func):
    """Cache with Streamlit when available, else a plain pass-through."""
    try:
        import streamlit as st

        return st.cache_data(ttl=CACHE_SECONDS)(func)
    except ImportError:
        return func


@_cached
def load_locations():
    """Return the 17 NCR sample points (location_id, city, lat, lng, elevation)."""
    return _query(LOCATIONS_SQL, [])


@_cached
def load_recent_flood_features(days: int = DEFAULT_DAYS, city: str | None = None):
    """Return the last ``days`` of flood features, one row per city per day.

    Columns: date, city, rain_mm, rain_max_1h_mm, rain_max_3h_mm,
    rain_max_6h_mm, rain_3d_mm, rain_7d_mm, rain_14d_mm, wet_streak_days,
    river_discharge_m3s. Pass ``city`` (e.g. ``"marikina"``) to keep one city.
    """
    sql, params = FLOOD_FEATURES_SQL, [_clamp_days(days)]
    if city:
        # Insert the city filter before GROUP BY.
        sql = sql.replace("GROUP BY", CITY_FILTER + "\nGROUP BY", 1)
        params.append(city.strip().lower())
    return _query(sql, params)


@_cached
def load_recent_weather(days: int = DEFAULT_DAYS, city: str | None = None):
    """Return the last ``days`` of daily weather, one row per city per day.

    Columns: date, city, rain_mm, temp_max_c, temp_min_c, wind_max_kmh,
    wind_gust_kmh, river_discharge_m3s. Pass ``city`` to keep one city.
    """
    sql, params = WEATHER_SQL, [_clamp_days(days)]
    if city:
        sql = sql.replace("GROUP BY", CITY_FILTER + "\nGROUP BY", 1)
        params.append(city.strip().lower())
    return _query(sql, params)


def load_7day_overview(city: str | None = None):
    """Return the default 7-day view for the Streamlit home screen.

    Returns a dict with ``flood_features`` and ``weather`` DataFrames.
    """
    return {
        "flood_features": load_recent_flood_features(DEFAULT_DAYS, city),
        "weather": load_recent_weather(DEFAULT_DAYS, city),
    }


if __name__ == "__main__":
    overview = load_7day_overview()
    for name, frame in overview.items():
        print(f"{name}: {len(frame)} rows")
        if not frame.empty:
            print(frame.head().to_string(index=False))
