"""Fetch Open-Meteo weather data that matches a user's question.

The weather agent passes in the user's question (English, Filipino or a mix).
This module works out three things from it:

    1. Where: which Metro Manila city (defaults to Manila).
    2. What:  which kinds of data are relevant (rain, wind, heat, ...).
    3. When:  how many days ahead to look.

It then requests only the matching variables from Open-Meteo, so the agent
receives focused data instead of everything the API offers.

Example:
    data = fetch_open_meteo("Uulan ba bukas sa Quezon City?")
    data["plan"]     # what was detected: city, topics, days
    data["weather"]  # raw Open-Meteo response with only rain-related fields
"""

import httpx

from utils.ttl_cache import ttl_cache

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Manila"
REQUEST_TIMEOUT_SECONDS = 10.0
FORECAST_CACHE_SECONDS = 30 * 60  # demo setting; Open-Meteo itself updates every 15 minutes

DEFAULT_CITY = "manila"
DEFAULT_FORECAST_DAYS = 2

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


def fetch_open_meteo(question: str, client: httpx.Client | None = None) -> dict:
    """Fetch the Open-Meteo data that matches the question.

    Args:
        question: The user's question in English, Filipino, or mixed.
        client: Optional HTTP client, mainly so tests can avoid real requests.

    Returns:
        {"plan": <what was detected>, "weather": <Open-Meteo JSON response>}

    Raises:
        httpx.HTTPError: if the request fails or Open-Meteo returns an error.
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

    return {"plan": plan, "weather": weather}


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
    # Quick manual check: python -m agents.fetch_open_meteo "Uulan ba bukas sa Makati?"
    import json
    import sys

    sample_question = " ".join(sys.argv[1:]) or "Will it rain tomorrow in Quezon City?"
    print(json.dumps(fetch_open_meteo(sample_question), indent=2))
