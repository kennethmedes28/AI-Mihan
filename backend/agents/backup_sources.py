"""Backup sources, used only when the PAGASA bulletin is empty or cannot be trusted.

The order of trust is:

    1. PAGASA bulletin           (checked by ``is_pagasa_reliable``)
    2. Backup sources, each read on its own (this file):
         * tide height   - Open-Meteo marine API (high tide slows drainage)
         * ground height - Open-Meteo elevation API (low ground floods first)
         * official sites - NDRRMC, Red Cross, DSWD (see ``web_lookup_agent``)
    3. The fixed safe answer, when too few readings are left to say anything
       responsible (see ``enough_readings``)

Each backup is read independently. One that fails is logged and skipped; it
never stops the others. Nothing here is guessed: a reading is either real data
or ``None``.

Example:
    backups = gather_backup_sources("marikina", now)
    if enough_readings(backups, readings_from_elsewhere=1):
        ...
"""

import contextvars
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import httpx

from agents.web_lookup_agent import search_trusted_sites
from src.rag.models import PHILIPPINE_TZ
from utils.ttl_cache import ttl_cache

logger = logging.getLogger(__name__)

MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
ELEVATION_API_URL = "https://api.open-meteo.com/v1/elevation"
REQUEST_TIMEOUT_SECONDS = 10.0
CACHE_SECONDS = 30 * 60
TIMEZONE = "Asia/Manila"

# --- What counts as a reliable PAGASA bulletin ------------------------------

MIN_PAGASA_CHARS = 200  # a real bulletin is far longer; shorter text is a stub or an error page

# Phrases that show the page we got was a block or error page, not a bulletin.
ERROR_PAGE_PHRASES = [
    "just a moment",
    "access denied",
    "403 forbidden",
    "404 not found",
    "page not found",
    "service unavailable",
    "bad gateway",
]

# A bulletin issued longer ago than this is out of date. PAGASA issues forecasts at least twice a day and
# cyclone bulletins every few hours, so a day-old bulletin no longer describes the weather.
MAX_BULLETIN_AGE = timedelta(hours=24)

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
MONTH_NAMES = "|".join(MONTHS)
# Words that introduce the issue time in a bulletin: "Issued at 5:00 PM, 31 October 2026", "as of ...".
ISSUE_MARKER = re.compile(r"\b(?:issued|as of|updated)\b[^0-9a-z]{0,6}(?:at|on)?", re.IGNORECASE)
DAY_FIRST_DATE = re.compile(rf"(\d{{1,2}})\s+({MONTH_NAMES})\s+(\d{{4}})", re.IGNORECASE)
MONTH_FIRST_DATE = re.compile(rf"({MONTH_NAMES})\s+(\d{{1,2}}),?\s+(\d{{4}})", re.IGNORECASE)
CLOCK_TIME = re.compile(r"(\d{1,2}):(\d{2})\s*([ap])\.?m", re.IGNORECASE)
ISSUE_WINDOW_CHARS = 60  # how far after the marker the date and time are looked for

# --- Rules of thumb for the backup readings (tune against past floods) ------

HIGH_TIDE_M = 1.2  # metres above mean sea level; Manila Bay peaks near 1.5 m on spring tides
LOW_GROUND_M = 5.0  # metres above sea level; at or below this, water drains out slowly

# Without PAGASA, at least this many separate readings must be available
# (rain forecast, river flow, tide, official web pages) before we give a level.
MIN_READINGS_WITHOUT_PAGASA = 2


@dataclass
class BackupData:
    """What the backup sources found. A field left empty means that source gave nothing."""

    peak_tide_m: float | None = None  # highest tide in the next 24 hours
    elevation_m: float | None = None
    web_pages: list[dict] = field(default_factory=list)

    @property
    def is_high_tide(self) -> bool:
        return self.peak_tide_m is not None and self.peak_tide_m >= HIGH_TIDE_M

    @property
    def is_low_ground(self) -> bool:
        return self.elevation_m is not None and self.elevation_m <= LOW_GROUND_M

    def readings_found(self) -> int:
        """How many separate readings (tide, official pages) this holds. Elevation does not count:
        it is fixed ground height, not a sign of what the weather is doing now."""
        return int(self.peak_tide_m is not None) + int(bool(self.web_pages))


# ---------------------------------------------------------------- PAGASA check


def find_issue_time(text: str) -> datetime | None:
    """The newest issue time written in a bulletin ("Issued at 5:00 PM, 31 October 2026"), in Manila time.

    Only dates that follow "issued", "as of" or "updated" count, so a date mentioned in the body
    (a past typhoon) is not mistaken for the issue time. A date with no clock time counts as the
    end of that day, which errs towards trusting the bulletin. Returns None when no issue time is written.
    """
    found: list[datetime] = []
    for marker in ISSUE_MARKER.finditer(text):
        window = text[marker.start(): marker.end() + ISSUE_WINDOW_CHARS]
        date_match = DAY_FIRST_DATE.search(window)
        if date_match:
            day, month, year = date_match.groups()
        else:
            date_match = MONTH_FIRST_DATE.search(window)
            if not date_match:
                continue
            month, day, year = date_match.groups()

        hour, minute = 23, 59
        clock = CLOCK_TIME.search(window)
        if clock:
            hour, minute = int(clock.group(1)) % 12, int(clock.group(2))
            if clock.group(3).lower() == "p":
                hour += 12
        try:
            found.append(datetime(int(year), MONTHS.index(month.lower()) + 1, int(day), hour, minute, tzinfo=PHILIPPINE_TZ))
        except ValueError:  # "31 February" or an hour that does not exist: not a real date
            continue
    return max(found) if found else None


def is_bulletin_stale(text: str, now: datetime | None = None) -> bool:
    """True when the bulletin says it was issued more than ``MAX_BULLETIN_AGE`` ago.

    A bulletin that writes no issue time cannot be judged and is not called stale.
    """
    issued = find_issue_time(text)
    now = now or datetime.now(PHILIPPINE_TZ)
    return issued is not None and now - issued > MAX_BULLETIN_AGE


def is_usable_bulletin(page: dict, now: datetime | None = None) -> bool:
    """True when one PAGASA page is long enough, is not a block/error page, and is not out of date."""
    text = page.get("text", "")
    lowered = text.lower()
    return (
        len(text.strip()) >= MIN_PAGASA_CHARS
        and not any(phrase in lowered for phrase in ERROR_PAGE_PHRASES)
        and not is_bulletin_stale(text, now)
    )


def is_pagasa_reliable(pages: list[dict] | None, now: datetime | None = None) -> bool:
    """True when the PAGASA pages look like a real, current bulletin.

    Empty, very short, or block/error pages (for example a "Just a moment..."
    challenge page) are not reliable, and neither is a bulletin whose issue
    time is more than a day old.
    """
    return any(is_usable_bulletin(page, now) for page in pages or [])


def pagasa_missing_reason(pages: list[dict] | None, now: datetime | None = None) -> str:
    """Name for the missing bulletin in an answer: "out of date" when a long, clean page was too old."""
    for page in pages or []:
        text = page.get("text", "")
        looks_real = len(text.strip()) >= MIN_PAGASA_CHARS and not any(
            phrase in text.lower() for phrase in ERROR_PAGE_PHRASES
        )
        if looks_real and is_bulletin_stale(text, now):
            return "PAGASA bulletin (out of date)"
    return "PAGASA bulletin"


# ---------------------------------------------------------------- the backup readings


@ttl_cache(ttl_seconds=CACHE_SECONDS)
def download_tide(lat: float, lng: float) -> dict:
    """Hourly sea level (m, tide included) for the next 2 days, from the Open-Meteo marine API."""
    params = {
        "latitude": lat,
        "longitude": lng,
        "hourly": "sea_level_height_msl",
        "forecast_days": 2,
        "timezone": TIMEZONE,
    }
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.get(MARINE_API_URL, params=params)
        response.raise_for_status()
        return response.json()


@ttl_cache(ttl_seconds=CACHE_SECONDS)
def download_elevation(lat: float, lng: float) -> dict:
    """Ground height (m above sea level) at a point, from the Open-Meteo elevation API."""
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.get(ELEVATION_API_URL, params={"latitude": lat, "longitude": lng})
        response.raise_for_status()
        return response.json()


def fetch_peak_tide(lat: float, lng: float, now: datetime) -> float | None:
    """Highest sea level (m) in the next 24 hours, or None when there is no tide data here.

    The marine model only covers the sea, so a point far inland can come back
    with empty values. That is "no data", not "no tide".
    """
    try:
        hourly = download_tide(lat, lng)["hourly"]
        now_manila = now.astimezone(PHILIPPINE_TZ).replace(tzinfo=None)
        upcoming = [
            height
            for time_text, height in zip(hourly["time"], hourly["sea_level_height_msl"])
            if height is not None and 0 < (datetime.fromisoformat(time_text) - now_manila).total_seconds() <= 24 * 3600
        ]
        return max(upcoming) if upcoming else None
    except Exception as error:
        logger.warning("Tide data unavailable: %s", error)
        return None


def fetch_elevation(lat: float, lng: float) -> float | None:
    """Ground height in metres, or None when it could not be read."""
    try:
        return float(download_elevation(lat, lng)["elevation"][0])
    except Exception as error:
        logger.warning("Elevation data unavailable: %s", error)
        return None


def fetch_official_pages(city: str) -> list[dict]:
    """Recent pages from the trusted official sites about flooding in the city (may be empty)."""
    try:
        return search_trusted_sites(f"flood rainfall warning {city}")
    except Exception as error:  # Never let a backup break the answer.
        logger.warning("Official site search failed: %s", error)
        return []


def gather_backup_sources(city: str, lat: float, lng: float, now: datetime) -> BackupData:
    """Read every backup source on its own and collect what was found.

    The sources do not depend on each other, so they are read at the same time.
    """
    with ThreadPoolExecutor(max_workers=3) as executor:
        # copy_context() carries the LangSmith run into the worker threads.
        tide = executor.submit(contextvars.copy_context().run, fetch_peak_tide, lat, lng, now)
        elevation = executor.submit(contextvars.copy_context().run, fetch_elevation, lat, lng)
        pages = executor.submit(contextvars.copy_context().run, fetch_official_pages, city)
        return BackupData(
            peak_tide_m=tide.result(),
            elevation_m=elevation.result(),
            web_pages=pages.result(),
        )


def enough_readings(backups: BackupData, readings_from_elsewhere: int) -> bool:
    """True when enough separate readings exist to answer without PAGASA.

    ``readings_from_elsewhere`` is how many of the rain forecast and river flow
    were read (0, 1 or 2).
    """
    return backups.readings_found() + readings_from_elsewhere >= MIN_READINGS_WITHOUT_PAGASA
