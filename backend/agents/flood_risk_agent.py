"""Flood risk agent: turns rain, river and PAGASA data into one risk level per city.

The level is worked out by fixed rules (no chat model), so it is the same
every time for the same data and costs nothing per question:

    low -> watch -> high -> critical

Inputs, all optional (a missing one is simply skipped and said in the answer):

    * Open-Meteo hourly rain, past 3 days and next 24 hours
    * Open-Meteo river discharge forecast
    * PAGASA rainfall warning colour, if the caller passes the PAGASA text
    * the project's daily dataset, to know what a "high" river or "wet" ground is for each city

When the PAGASA bulletin is empty or unreliable, the backup sources in
``backup_sources`` (tide, ground height, official sites) are read as well. If too
few readings are left to be responsible, the fixed safe answer is returned
instead of a level that may be too low.

The thresholds are rules of thumb. Tune them against past floods.

Example:
    risk = assess_flood_risk("marikina")
    print(risk.level, risk.reasons)
"""

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import httpx

from agents.fetch_supabase_open_meteo_datasets import NCR_LOCATIONS, TIMEZONE
from agents.flood_types import (
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_ORDER,
    RISK_WATCH,
    AgentReply,
    ChatContext,
)
from agents.backup_sources import BackupData, enough_readings, gather_backup_sources, is_pagasa_reliable
from agents.location_agent import ASK_FOR_LOCATION, find_city
from agents.safe_answer import safe_reply
from src.rag.models import PHILIPPINE_TZ
from utils.ttl_cache import ttl_cache
from utils.tracing import traced_agent

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FLOOD_API_URL = "https://flood-api.open-meteo.com/v1/flood"
REQUEST_TIMEOUT_SECONDS = 10.0
CACHE_SECONDS = 30 * 60
DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "ncr_flood_features_daily.csv"

# Peak rain in one hour (mm/h), based on PAGASA's yellow / orange / red rainfall warnings.
HOURLY_RAIN_WATCH_MM = 7.5
HOURLY_RAIN_HIGH_MM = 15.0
HOURLY_RAIN_CRITICAL_MM = 30.0

# Total rain in the next 24 hours (mm).
DAILY_RAIN_WATCH_MM = 40.0
DAILY_RAIN_HIGH_MM = 75.0
DAILY_RAIN_CRITICAL_MM = 150.0

SATURATION_PERCENTILE = 0.90  # past 3-day rain above this share of history = soaked ground
RIVER_PERCENTILE = 0.95  # forecast river flow above this share of history = high river

# PAGASA rainfall warning words, strongest first.
PAGASA_WARNINGS = [
    ("red rainfall warning", RISK_CRITICAL),
    ("orange rainfall warning", RISK_HIGH),
    ("yellow rainfall warning", RISK_WATCH),
]

LEVEL_ADVICE = {
    RISK_LOW: (
        "No flood signal right now. Keep checking pagasa.dost.gov.ph.",
        "Walang senyales ng baha sa ngayon. Patuloy na tingnan ang pagasa.dost.gov.ph.",
    ),
    RISK_WATCH: (
        "Flooding is possible in low areas. Prepare your go bag and know your nearest evacuation center.",
        "Maaaring bumaha sa mabababang lugar. Ihanda ang go bag at alamin ang pinakamalapit na evacuation center.",
    ),
    RISK_HIGH: (
        "Flooding is likely. Move valuables up, avoid low roads, and be ready to leave.",
        "Malamang na bumaha. Ilipat sa mataas ang gamit, iwasan ang mababang kalsada, at maghanda sa paglikas.",
    ),
    RISK_CRITICAL: (
        "Serious flooding is expected. Follow your DRRMO's evacuation orders now.",
        "Inaasahan ang malubhang baha. Sundin agad ang utos ng inyong DRRMO sa paglikas.",
    ),
}


@dataclass
class FloodRisk:
    """The risk level for one city, with the reasons behind it."""

    city: str
    level: str
    reasons: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # data sources that could not be read
    as_of: datetime | None = None


def raise_level(level: str, steps: int = 1) -> str:
    """Move a level up by ``steps``, stopping at critical."""
    return RISK_ORDER[min(RISK_ORDER.index(level) + steps, len(RISK_ORDER) - 1)]


def higher_of(first: str, second: str) -> str:
    """The more serious of two levels."""
    return max(first, second, key=RISK_ORDER.index)


# ---------------------------------------------------------------- history from the dataset


@lru_cache(maxsize=1)
def load_history() -> dict[str, dict[str, list[float]]]:
    """Read the daily dataset once: ``{city: {"rain_3d": sorted values, "discharge": sorted values}}``.

    The dataset's ``location_id`` is the position of the city in ``NCR_LOCATIONS``.
    Returns an empty dict if the file is missing.
    """
    path = Path(os.getenv("FLOOD_DATASET", DEFAULT_DATASET))
    if not path.exists():
        logger.warning("Flood dataset not found at %s; history rules are skipped.", path)
        return {}

    cities = list(NCR_LOCATIONS)
    history = {city: {"rain_3d": [], "discharge": []} for city in cities}
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                city = cities[int(row["location_id"])]
            except (ValueError, IndexError):
                continue
            for column, key in (("rain_3d_mm", "rain_3d"), ("river_discharge_m3s", "discharge")):
                if row.get(column):
                    history[city][key].append(float(row[column]))

    for values in history.values():
        values["rain_3d"].sort()
        values["discharge"].sort()
    return history


def percentile_value(sorted_values: list[float], share: float) -> float | None:
    """The value below which ``share`` (0..1) of the history lies."""
    if not sorted_values:
        return None
    return sorted_values[min(int(share * len(sorted_values)), len(sorted_values) - 1)]


# ---------------------------------------------------------------- live data


@ttl_cache(ttl_seconds=CACHE_SECONDS)
def download_rain(lat: float, lng: float) -> dict:
    """Hourly rain (mm) for the past 3 days and the next 2, from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lng,
        "hourly": "precipitation",
        "past_days": 3,
        "forecast_days": 2,
        "timezone": TIMEZONE,
    }
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.get(FORECAST_URL, params=params)
        response.raise_for_status()
        return response.json()


@ttl_cache(ttl_seconds=CACHE_SECONDS)
def download_river(lat: float, lng: float) -> dict:
    """Daily river flow (m3/s) forecast for the next 3 days, from Open-Meteo."""
    params = {"latitude": lat, "longitude": lng, "daily": "river_discharge", "forecast_days": 3}
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.get(FLOOD_API_URL, params=params)
        response.raise_for_status()
        return response.json()


def split_rain(rain: dict, now: datetime) -> tuple[list[float], list[float]]:
    """Split hourly rain into (past 72 hours, next 24 hours) around ``now`` (Manila time)."""
    times = rain["hourly"]["time"]
    amounts = [value or 0.0 for value in rain["hourly"]["precipitation"]]
    now_naive = now.astimezone(PHILIPPINE_TZ).replace(tzinfo=None)

    past, future = [], []
    for time_text, amount in zip(times, amounts):
        hour = datetime.fromisoformat(time_text)
        if hour <= now_naive:
            past.append((hour, amount))
        elif (hour - now_naive).total_seconds() <= 24 * 3600:
            future.append(amount)
    return [amount for _, amount in past[-72:]], future


# ---------------------------------------------------------------- the rules


def assess_flood_risk(
    city: str,
    pagasa_text: str = "",
    now: datetime | None = None,
    rain: dict | None = None,
    river: dict | None = None,
) -> FloodRisk:
    """Work out the flood risk level for a city.

    ``rain`` and ``river`` are the downloaded Open-Meteo answers; when left
    out they are downloaded (tests pass them in). A source that fails is
    listed in ``missing`` and the other sources still count.
    """
    now = now or datetime.now(PHILIPPINE_TZ)
    lat, lng = NCR_LOCATIONS[city]
    risk = FloodRisk(city=city, level=RISK_LOW, as_of=now)
    history = load_history().get(city, {})

    # 1. Official PAGASA rainfall warning colour.
    lowered_text = pagasa_text.lower()
    for phrase, level in PAGASA_WARNINGS:
        if phrase in lowered_text:
            risk.level = higher_of(risk.level, level)
            risk.reasons.append(f"PAGASA has a {phrase}.")
            break

    # 2. Rain forecast and how wet the ground already is.
    try:
        rain = rain or download_rain(lat, lng)
        past_hours, next_hours = split_rain(rain, now)
    except Exception as error:
        logger.warning("Rain data unavailable for %s: %s", city, error)
        risk.missing.append("rain forecast")
        past_hours, next_hours = [], []

    if next_hours:
        peak, total = max(next_hours), sum(next_hours)
        for limit, level in (
            (HOURLY_RAIN_CRITICAL_MM, RISK_CRITICAL),
            (HOURLY_RAIN_HIGH_MM, RISK_HIGH),
            (HOURLY_RAIN_WATCH_MM, RISK_WATCH),
        ):
            if peak >= limit:
                risk.level = higher_of(risk.level, level)
                risk.reasons.append(f"Forecast peak rain {peak:.1f} mm in one hour (next 24 h).")
                break
        for limit, level in (
            (DAILY_RAIN_CRITICAL_MM, RISK_CRITICAL),
            (DAILY_RAIN_HIGH_MM, RISK_HIGH),
            (DAILY_RAIN_WATCH_MM, RISK_WATCH),
        ):
            if total >= limit:
                risk.level = higher_of(risk.level, level)
                risk.reasons.append(f"Forecast total rain {total:.0f} mm in the next 24 hours.")
                break

    saturated_from = percentile_value(history.get("rain_3d", []), SATURATION_PERCENTILE)
    if past_hours and saturated_from is not None and sum(past_hours) >= saturated_from:
        risk.reasons.append(f"The ground is already soaked ({sum(past_hours):.0f} mm in the past 3 days).")
        if risk.level != RISK_LOW:
            risk.level = raise_level(risk.level)
        else:
            risk.level = RISK_WATCH

    # 3. River flow compared with this city's history.
    try:
        river = river or download_river(lat, lng)
        flows = [value for value in river["daily"]["river_discharge"] if value is not None]
    except Exception as error:
        logger.warning("River data unavailable for %s: %s", city, error)
        risk.missing.append("river flow")
        flows = []

    high_flow = percentile_value(history.get("discharge", []), RIVER_PERCENTILE)
    if flows and high_flow is not None and max(flows) >= high_flow:
        risk.reasons.append(f"River flow is forecast to reach {max(flows):.1f} m3/s, high for this area.")
        risk.level = higher_of(risk.level, RISK_HIGH)

    if not risk.reasons and not risk.missing:
        risk.reasons.append("No heavy rain, soaked ground or high river flow in the data.")
    return risk


def format_risk(risk: FloodRisk) -> str:
    """Write the risk as a short bilingual message."""
    english, filipino = LEVEL_ADVICE[risk.level]
    when = risk.as_of.astimezone(PHILIPPINE_TZ).strftime("%b %d, %I:%M %p") if risk.as_of else "unknown time"
    lines = [f"Flood risk for {risk.city.title()}: {risk.level.upper()} (as of {when})"]
    lines += [f"- {reason}" for reason in risk.reasons]
    if risk.missing:
        lines.append(f"- Could not read: {', '.join(risk.missing)}. The level may be too low.")
    lines += [english, filipino, "This is a rule-based estimate, not an official warning. Official: pagasa.dost.gov.ph"]
    return "\n".join(lines)


def apply_backup_readings(risk: FloodRisk, backups: BackupData) -> None:
    """Add what the backup sources found to the risk, as reasons.

    A high tide or low ground can only make an existing flood signal one step
    worse (water cannot drain away). On their own they never create a signal,
    so a calm forecast stays "low".
    """
    if backups.is_high_tide:
        risk.reasons.append(f"High tide is forecast ({backups.peak_tide_m:.1f} m), so floodwater drains slowly.")
    if backups.is_low_ground:
        risk.reasons.append(f"This area is low ground ({backups.elevation_m:.0f} m above sea level).")
    if (backups.is_high_tide or backups.is_low_ground) and risk.level != RISK_LOW:
        risk.level = raise_level(risk.level)


def format_web_updates(pages: list[dict]) -> str:
    """Links to recent official pages about the city, for when the PAGASA bulletin could not be read.

    Their text is NOT used to set the risk level: a web search can return an old
    article, and an old "red warning" must not raise a false alarm. The user gets
    the links to read for themselves.
    """
    if not pages:
        return ""
    lines = [f"- {page['title'] or page['source']}: {page['source']}" for page in pages[:3]]
    return "\n\nFrom a web search of official sites (not used in the level above):\n" + "\n".join(lines)


@traced_agent("flood_risk_agent")
def answer_flood_risk_question(question: str, context: ChatContext) -> AgentReply:
    """Answer "will it flood?" for the city in the question, or the user's saved city.

    Order: PAGASA first. If its bulletin is empty or unreliable, the backup
    sources are read too. If too few readings remain, the safe answer is returned.
    """
    city = find_city(question) or (context.location.city if context.location else None)
    if city is None:
        return AgentReply(text=ASK_FOR_LOCATION, pending_intent="flood_risk")

    try:
        from agents.fetch_weather_using_browser import fetch_weather_pages

        pages = fetch_weather_pages("flood rainfall warning")
    except Exception as error:  # PAGASA site down or blocked; the other sources still work.
        logger.warning("PAGASA pages unavailable: %s", error)
        pages = []

    pagasa_ok = is_pagasa_reliable(pages)
    pagasa_text = " ".join(page["text"] for page in pages) if pagasa_ok else ""
    risk = assess_flood_risk(city, pagasa_text, now=context.now)
    if pagasa_ok:
        return AgentReply(text=format_risk(risk))

    # PAGASA is empty or unreliable: use the backup sources, then decide if that is enough.
    risk.missing.append("PAGASA bulletin")
    now = context.now or datetime.now(PHILIPPINE_TZ)
    lat, lng = NCR_LOCATIONS[city]
    backups = gather_backup_sources(city, lat, lng, now)
    apply_backup_readings(risk, backups)

    # Rain and river were read unless assess_flood_risk listed them as missing.
    weather_readings = sum(name not in risk.missing for name in ("rain forecast", "river flow"))
    if not enough_readings(backups, weather_readings):
        logger.info("PAGASA unreliable and too few backup readings for %s; using the safe answer.", city)
        return safe_reply()

    return AgentReply(text=format_risk(risk) + format_web_updates(backups.web_pages))
