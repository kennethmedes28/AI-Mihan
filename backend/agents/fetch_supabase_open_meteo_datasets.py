"""Fetch the Open-Meteo data that matches a user's question.

The weather agent passes in the user's question (English, Filipino or a mix).
This module works out three things from it:

    1. Where: which Metro Manila city (defaults to Manila).
    2. What:  which kinds of data are relevant (rain, wind, heat, ...).
    3. When:  how many days ahead to look.

It then gets two kinds of data:

    * the live forecast, requesting only the matching variables from the
      Open-Meteo API, so the agent receives focused data instead of everything;
    * the last few days of rain and river discharge for that city from the
      Supabase dataset tables (kept current by scripts/update_open_meteo_datasets.py).

The history is extra context. If Supabase is down, or the city is not in the
dataset, the forecast is still returned.

Example:
    data = fetch_open_meteo("Uulan ba bukas sa Quezon City?")
    data["plan"]     # what was detected: city, topics, days
    data["weather"]  # raw Open-Meteo response with only rain-related fields
    data["history"]  # list of daily rows (oldest first), [] if the city has no
                     # dataset, or None if Supabase could not be read
"""

import logging

import httpx

from src.services.fetch_postgre_database import get_connection
from utils.tracing import traced_agent
from utils.ttl_cache import ttl_cache

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Manila"
REQUEST_TIMEOUT_SECONDS = 10.0
FORECAST_CACHE_SECONDS = 30 * 60  # demo setting; Open-Meteo itself updates every 15 minutes

DEFAULT_CITY = "manila"
DEFAULT_FORECAST_DAYS = 2

HISTORY_DAYS = 7  # how many past days to read from Supabase
HISTORY_CACHE_SECONDS = 30 * 60  # the tables only change once a day

# One row per day for a city. A city can have several sample points, so rain
# is averaged and the discharge / wet streak take the highest point. The
# city name is compared without the accent ("Las Piñas" = "las pinas") and
# without a trailing note ("Pateros (municipality)" = "pateros").
HISTORY_SQL = """
SELECT f.date,
       ROUND(AVG(f.precipitation_sum_mm)::numeric, 1)::float AS rain_mm,
       ROUND(AVG(f.rain_max_1h_mm)::numeric, 1)::float       AS rain_max_1h_mm,
       ROUND(AVG(f.rain_3d_mm)::numeric, 1)::float           AS rain_3d_mm,
       ROUND(AVG(f.rain_7d_mm)::numeric, 1)::float           AS rain_7d_mm,
       ROUND(AVG(f.rain_14d_mm)::numeric, 1)::float          AS rain_14d_mm,
       MAX(f.wet_streak_days)                                AS wet_streak_days,
       MAX(f.river_discharge_m3s)::float                     AS river_discharge_m3s
FROM ncr_flood_features_daily f
JOIN ncr_locations l USING (location_id)
WHERE translate(lower(regexp_replace(l.city, ' \\(.*\\)$', '')), 'ñ', 'n') = %s
  AND f.date > (SELECT max(date) FROM ncr_flood_features_daily) - %s
GROUP BY f.date
ORDER BY f.date
"""

# Metro Manila cities: name -> (latitude, longitude).
NCR_LOCATIONS = {
    "manila": (14.5995, 120.9842),
    "quezon city": (14.6760, 121.0437),
    "makati": (14.5547, 121.0244),
    "pasig": (14.5764, 121.0851),
    "taguig": (14.5176, 121.0509),
    "marikina": (14.6507, 121.1029),
    "mandaluyong": (14.5794, 121.0359),
    "san juan": (14.6019, 121.0355),
    "pasay": (14.5378, 121.0014),
    "paranaque": (14.4793, 121.0198),
    "las pinas": (14.4445, 120.9939),
    "muntinlupa": (14.4081, 121.0415),
    "caloocan": (14.6488, 120.9830),
    "malabon": (14.6625, 120.9568),
    "navotas": (14.6667, 120.9417),
    "valenzuela": (14.7011, 120.9830),
    "pateros": (14.5454, 121.0687),
}

# Other spellings people commonly use -> the key used above.
LOCATION_ALIASES = {
    "qc": "quezon city",
    "parañaque": "paranaque",
    "las piñas": "las pinas",
    "maynila": "manila",
    "metro manila": "manila",
    "ncr": "manila",
}

# Each topic lists the words that trigger it and the Open-Meteo variables it needs.
TOPICS = {
    "rain": {
        "keywords": ["rain", "ulan", "uulan", "shower", "flood", "baha", "umbrella", "payong"],
        "current": ["precipitation"],
        "hourly": ["precipitation", "precipitation_probability"],
        "daily": ["precipitation_sum", "precipitation_probability_max"],
    },
    "wind": {
        "keywords": ["wind", "hangin", "gust", "malakas na hangin", "signal"],
        "current": ["wind_speed_10m", "wind_gusts_10m"],
        "hourly": ["wind_speed_10m", "wind_gusts_10m"],
        "daily": ["wind_speed_10m_max", "wind_gusts_10m_max"],
    },
    "heat": {
        "keywords": ["hot", "heat", "init", "temperature", "temperatura", "mainit", "lamig", "cold"],
        "current": ["temperature_2m", "apparent_temperature", "relative_humidity_2m"],
        "hourly": ["temperature_2m", "apparent_temperature"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_max"],
    },
    "storm": {
        "keywords": ["typhoon", "bagyo", "storm", "cyclone", "habagat", "low pressure"],
        # A falling pressure reading is an early storm sign; also needs rain and wind.
        "current": ["pressure_msl", "precipitation", "wind_speed_10m", "wind_gusts_10m"],
        "hourly": ["pressure_msl", "precipitation", "wind_gusts_10m"],
        "daily": ["precipitation_sum", "wind_gusts_10m_max"],
    },
}

# Used when the question matches no topic (e.g. "How is the weather?").
GENERAL_VARIABLES = {
    "current": ["temperature_2m", "precipitation", "wind_speed_10m", "weather_code"],
    "hourly": ["temperature_2m", "precipitation_probability"],
    "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
}

# Words that change how far ahead to look. Checked from longest range to shortest.
FORECAST_RANGE_KEYWORDS = [
    (7, ["week", "linggo", "7 days", "7 araw"]),
    (3, ["3 days", "3 araw", "weekend", "next few days"]),
    (2, ["tomorrow", "bukas", "next day"]),
    (1, ["today", "ngayon", "tonight", "mamaya"]),
]


def detect_city(question: str) -> str:
    """Return the NCR city named in the question, or the default city."""
    text = question.lower()

    for alias, city in LOCATION_ALIASES.items():
        if alias in text:
            return city

    # Match longer names first so "san juan" is not confused with a shorter name.
    for city in sorted(NCR_LOCATIONS, key=len, reverse=True):
        if city in text:
            return city

    return DEFAULT_CITY


def detect_topics(question: str) -> list[str]:
    """Return the topics (rain, wind, ...) the question is about."""
    text = question.lower()
    return [
        topic
        for topic, spec in TOPICS.items()
        if any(keyword in text for keyword in spec["keywords"])
    ]


def detect_forecast_days(question: str) -> int:
    """Return how many forecast days the question needs."""
    text = question.lower()
    for days, keywords in FORECAST_RANGE_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return days
    return DEFAULT_FORECAST_DAYS


def build_plan(question: str) -> dict:
    """Decide what to fetch for a question, without calling the API.

    Returns a dict with the city, coordinates, topics, forecast days, and the
    de-duplicated Open-Meteo variable lists for current / hourly / daily data.
    """
    city = detect_city(question)
    topics = detect_topics(question)
    days = detect_forecast_days(question)

    # Fall back to a general summary when no topic matched.
    selected = [TOPICS[topic] for topic in topics] or [GENERAL_VARIABLES]

    variables = {}
    for kind in ("current", "hourly", "daily"):
        names = []
        for spec in selected:
            for name in spec[kind]:
                if name not in names:
                    names.append(name)
        variables[kind] = names

    latitude, longitude = NCR_LOCATIONS[city]
    return {
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "topics": topics or ["general"],
        "forecast_days": days,
        "variables": variables,
    }


@traced_agent("open_meteo_fetch", run_type="tool")
def fetch_open_meteo(
    question: str, client: httpx.Client | None = None, include_history: bool = True
) -> dict:
    """Fetch the forecast (and recent history) that matches the question.

    Args:
        question: The user's question in English, Filipino, or mixed.
        client: Optional HTTP client, mainly so tests can avoid real requests.
        include_history: Set to False to skip the Supabase read.

    Returns:
        {"plan": <what was detected>,
         "weather": <Open-Meteo JSON response>,
         "history": <recent daily rows, [] or None, see ``fetch_recent_history``>}

    Raises:
        httpx.HTTPError: if the forecast request fails or Open-Meteo returns an error.
            A failed history read never raises; it gives ``"history": None``.
    """
    plan = build_plan(question)

    request_args = (
        plan["latitude"],
        plan["longitude"],
        ",".join(plan["variables"]["current"]),
        ",".join(plan["variables"]["hourly"]),
        ",".join(plan["variables"]["daily"]),
        plan["forecast_days"],
    )

    # Different questions that need the same data (same city, variables, days)
    # share one cached download. A caller-supplied client skips the cache.
    if client is None:
        weather = download_forecast(*request_args)
    else:
        weather = download_forecast.__wrapped__(*request_args, client=client)

    history = fetch_recent_history(plan["city"]) if include_history else None
    return {"plan": plan, "weather": weather, "history": history}


def fetch_recent_history(city: str, days: int = HISTORY_DAYS) -> list[dict] | None:
    """Return the last ``days`` days of rain and discharge for a city, oldest first.

    Returns ``[]`` when the dataset has no sample point in that city (it covers
    7 of the NCR cities), and ``None`` when Supabase could not be read.
    """
    try:
        return download_history(city, days)
    except Exception as error:  # the forecast is still useful without the history
        logger.warning("Supabase history unavailable for %s: %s", city, error)
        return None


@ttl_cache(ttl_seconds=HISTORY_CACHE_SECONDS)
def download_history(city: str, days: int) -> list[dict]:
    """Read the daily rows from Supabase. Results are cached for a few minutes."""
    with get_connection() as conn:
        rows = conn.execute(HISTORY_SQL, (city, days)).fetchall()

    for row in rows:
        row["date"] = row["date"].isoformat()
    return rows


@ttl_cache(ttl_seconds=FORECAST_CACHE_SECONDS)
def download_forecast(
    latitude: float,
    longitude: float,
    current: str,
    hourly: str,
    daily: str,
    forecast_days: int,
    client: httpx.Client | None = None,
) -> dict:
    """Request one forecast from Open-Meteo. Results are cached for a few minutes."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": current,
        "hourly": hourly,
        "daily": daily,
        "forecast_days": forecast_days,
        "timezone": TIMEZONE,
    }

    # Use the caller's client if given, otherwise open a short-lived one.
    if client is None:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as new_client:
            response = new_client.get(OPEN_METEO_URL, params=params)
    else:
        response = client.get(OPEN_METEO_URL, params=params)

    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    # Quick manual check: python -m agents.fetch_supabase_open_meteo_datasets "Uulan ba bukas sa Makati?"
    import json
    import sys

    sample_question = " ".join(sys.argv[1:]) or "Will it rain tomorrow in Quezon City?"
    print(json.dumps(fetch_open_meteo(sample_question), indent=2))
