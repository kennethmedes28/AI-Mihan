"""Location agent: works out where the user is (or where they want to go).

Order of attempts, most exact first:

    1. coordinates typed or shared, such as "14.6507, 121.1029"
    2. "Brgy. <name>" plus a city, looked up with OpenStreetMap (Nominatim)
    3. a place named next to a city, such as "Gentry Manor Paranaque", looked up the same way
    4. a city name alone, which uses the centre of the city

The result says how exact it is (``precision``), so later answers can warn
when only the city centre is known. No chat model is used here.

Example:
    location = resolve_location("Nasa Brgy. Nangka, Marikina ako")
"""

import logging
import re
import unicodedata
from typing import Callable

import httpx

from agents.fetch_supabase_open_meteo_datasets import LOCATION_ALIASES, NCR_LOCATIONS
from agents.flood_types import (
    PRECISION_BARANGAY,
    PRECISION_CITY,
    PRECISION_GPS,
    PRECISION_PLACE,
    AgentReply,
    ChatContext,
    Location,
)
from utils.geo import haversine_m
from utils.ttl_cache import ttl_cache
from utils.tracing import traced_agent

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "Flood-Gentic-flood-assistant/0.1"  # Nominatim requires an identifying agent
GEOCODE_CACHE_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 8.0

# A place found next to a city name must lie within this distance of that city's centre,
# so a lookup that matched a same-named place elsewhere is thrown away.
MAX_PLACE_DISTANCE_M = 10_000
MAX_PLACE_WORDS = 4  # a subdivision or landmark name is rarely longer

# Words that end a place name when reading the words around a city. They are the small
# connecting words, plus topic words ("baha", "weather") that are not part of a place name.
PLACE_STOP_WORDS = {
    "nasa", "sa", "ng", "ang", "ay", "ako", "ko", "po", "na", "at", "ngayon", "dito", "malapit",
    "in", "at", "near", "from", "to", "the", "i", "im", "am", "here", "now", "and", "my", "is",
    "baha", "bumabaha", "flood", "flooded", "ulan", "rain", "weather", "forecast", "risk",
    "evacuation", "evacuate", "center", "centre", "centers", "route", "hotline", "contacts",
    "saan", "where", "paano", "how", "nearest", "pinakamalapit", "bukas", "tomorrow", "today",
}

# Rough bounding box of Metro Manila: west, south, east, north.
NCR_BOUNDS = (120.90, 14.35, 121.20, 14.80)

ASK_FOR_LOCATION = (
    "Which city or barangay are you in? Example: \"Brgy. Nangka, Marikina\". "
    "You can also share your location pin.\n"
    "Saang lungsod o barangay kayo? Halimbawa: \"Brgy. Nangka, Marikina\". "
    "Maaari rin ninyong ibahagi ang inyong location pin."
)

# Asked once when only a city is known. The examples are written so the reply is recognised as a place.
ASK_FOR_PLACE = (
    "I found {city}, but not exactly where you are. Reply with your barangay or subdivision, "
    "for example \"Brgy. Tambo, {city}\" or \"Gentry Manor, {city}\", or share your location pin.\n"
    "Nakita ko ang {city}, pero hindi pa kung nasaan kayo. Ibigay ang barangay o subdivision, "
    "halimbawa \"Brgy. Tambo, {city}\" o \"Gentry Manor, {city}\", o ibahagi ang location pin."
)

COORDINATES_PATTERN = re.compile(r"(-?\d{1,2}\.\d{3,})\s*[,;\s]\s*(-?\d{2,3}\.\d{3,})")
BARANGAY_PATTERN = re.compile(
    r"(?:brgy\.?|barangay|bgy\.?)\s+([a-z0-9][a-z0-9 .\-']*?)(?=,|\.\s|\s+(?:sa|in|ako|at|na|ng|from|to)\b|$)"
)


def plain_text(text: str) -> str:
    """Lower-case text without accents, so "Parañaque" matches "paranaque"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def city_key(city: str) -> str:
    """Return the NCR city key for a city name or alias (or the cleaned name if unknown)."""
    text = plain_text(city).strip()
    return LOCATION_ALIASES.get(text, text)


def match_city(text: str) -> tuple[str, int, int] | None:
    """Find the NCR city named in the text. Returns ``(city, start, end)`` of the match in the
    accent-free lower-case text (see ``plain_text``), or None."""
    plain = plain_text(text)
    names = {plain_text(alias): city for alias, city in LOCATION_ALIASES.items()}
    names.update({city: city for city in NCR_LOCATIONS})

    # Longest names first, so "san juan" is tried before shorter names.
    for name in sorted(names, key=len, reverse=True):
        found = re.search(rf"\b{re.escape(name)}\b", plain)
        if found:
            return names[name], found.start(), found.end()
    return None


def find_city(text: str) -> str | None:
    """Return the NCR city named in the text, or None. Unlike ``detect_city`` there is no default."""
    match = match_city(text)
    return match[0] if match else None


def words_next_to_city(text: str) -> list[str]:
    """Candidate place names written next to the city, the words before it first.

    "nasa gentry manor paranaque tambo ako" gives ["gentry manor", "tambo"]. Reading stops at
    the first connecting or topic word (``PLACE_STOP_WORDS``) or at punctuation, and takes at
    most ``MAX_PLACE_WORDS`` words. Empty candidates are left out.
    """
    match = match_city(text)
    if match is None:
        return []
    _, start, end = match
    plain = plain_text(text)

    # Only the part of the sentence the city is in; words past a full stop are another thought.
    # A comma right next to the city ("Divisoria, Manila") joins the place to it, so it is dropped first.
    before = re.split(r"[,.;!?]", plain[:start].rstrip(" ,"))[-1].split()
    after = re.split(r"[,.;!?]", plain[end:].lstrip(" ,"))[0].split()

    before_words: list[str] = []
    for word in reversed(before):
        if word in PLACE_STOP_WORDS or len(before_words) == MAX_PLACE_WORDS:
            break
        before_words.insert(0, word)

    after_words: list[str] = []
    for word in after:
        if word in PLACE_STOP_WORDS or len(after_words) == MAX_PLACE_WORDS:
            break
        after_words.append(word)

    return [" ".join(words) for words in (before_words, after_words) if words]


def find_coordinates(text: str) -> tuple[float, float] | None:
    """Return (lat, lng) if the text holds coordinates inside Metro Manila."""
    match = COORDINATES_PATTERN.search(text)
    if not match:
        return None
    lat, lng = float(match.group(1)), float(match.group(2))
    west, south, east, north = NCR_BOUNDS
    return (lat, lng) if south <= lat <= north and west <= lng <= east else None


def find_barangay(text: str) -> str | None:
    """Return the barangay named like "Brgy. Nangka", or None."""
    match = BARANGAY_PATTERN.search(plain_text(text))
    return match.group(1).strip(" .") if match else None


def mentions_place(text: str) -> bool:
    """Cheap check (no network) for whether the text names a place that ``resolve_location`` can use."""
    return bool(find_coordinates(text) or find_city(text) or find_barangay(text))


def nearest_city(lat: float, lng: float) -> str:
    """The NCR city whose centre is closest to the point."""
    return min(NCR_LOCATIONS, key=lambda city: haversine_m(lat, lng, *NCR_LOCATIONS[city]))


def location_from_coordinates(lat: float, lng: float) -> Location:
    """Make a location from a GPS point (for example one sent by the app)."""
    city = nearest_city(lat, lng)
    return Location(
        name=f"{lat:.5f}, {lng:.5f}", lat=lat, lng=lng, city=city, precision=PRECISION_GPS
    )


@ttl_cache(ttl_seconds=GEOCODE_CACHE_SECONDS)
def search_nominatim(query: str) -> tuple[float, float] | None:
    """Look a place up in OpenStreetMap, limited to Metro Manila. Returns (lat, lng) or None
    when nothing matches. Network errors are raised, so ``ttl_cache`` does not keep them."""
    west, south, east, north = NCR_BOUNDS
    params = {
        "q": f"{query}, Metro Manila, Philippines",
        "format": "jsonv2",
        "limit": 1,
        "countrycodes": "ph",
        "viewbox": f"{west},{north},{east},{south}",
        "bounded": 1,
    }
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(NOMINATIM_URL, params=params)
        response.raise_for_status()
        results = response.json()
    return (float(results[0]["lat"]), float(results[0]["lon"])) if results else None


def geocode_place(query: str) -> tuple[float, float] | None:
    """Like ``search_nominatim``, but a failed request gives None. Only a real "no match" is
    remembered, so a timeout is tried again on the next message."""
    try:
        return search_nominatim(query)
    except Exception as error:  # Network down, rate limited, and so on.
        logger.warning("Geocoding failed for %r: %s", query, error)
        return None


def resolve_place_near_city(
    text: str, city: str, geocoder: Callable[[str], tuple[float, float] | None]
) -> Location | None:
    """Look up a subdivision, village or landmark written next to the city, such as "Gentry Manor".

    The words before the city are tried first, then the words after it. A result is only
    accepted when it lies within ``MAX_PLACE_DISTANCE_M`` of the city centre. Returns None
    when nothing fits, and the caller falls back to the city centre.
    """
    city_lat, city_lng = NCR_LOCATIONS[city]
    for phrase in words_next_to_city(text):
        found = geocoder(f"{phrase.title()}, {city.title()}")
        if found and haversine_m(city_lat, city_lng, *found) <= MAX_PLACE_DISTANCE_M:
            return Location(
                name=f"{phrase.title()}, {city.title()}",
                lat=found[0],
                lng=found[1],
                city=city,
                precision=PRECISION_PLACE,
            )
    return None


def resolve_location(
    text: str,
    geocoder: Callable[[str], tuple[float, float] | None] = geocode_place,
) -> Location | None:
    """Work out a location from free text. Returns None if no place is recognised.

    ``geocoder`` is only replaced by tests.
    """
    coordinates = find_coordinates(text)
    if coordinates:
        return location_from_coordinates(*coordinates)

    city = find_city(text)
    barangay = find_barangay(text)

    if barangay:
        query = f"Barangay {barangay.title()}" + (f", {city.title()}" if city else "")
        found = geocoder(query)
        if found:
            found_city = city or nearest_city(*found)
            return Location(
                name=f"Brgy. {barangay.title()}, {found_city.title()}",
                lat=found[0],
                lng=found[1],
                city=found_city,
                barangay=barangay.title(),
                precision=PRECISION_BARANGAY,
            )

    if city:
        place = resolve_place_near_city(text, city, geocoder)
        if place:
            return place
        lat, lng = NCR_LOCATIONS[city]
        return Location(name=city.title(), lat=lat, lng=lng, city=city, precision=PRECISION_CITY)
    return None


def describe_precision(location: Location) -> str:
    """One bilingual line warning when only the city centre is known."""
    if location.precision != PRECISION_CITY:
        return ""
    return (
        "(I only know your city, so distances are from the city centre. Share your barangay or pin for better results.)\n"
        "(Lungsod lang ang alam ko, kaya mula sa sentro ng lungsod ang layo. Ibigay ang barangay o pin para mas tumpak.)"
    )


@traced_agent("location_agent")
def answer_location_message(question: str, context: ChatContext) -> AgentReply:
    """Handle a message that only gives a place, such as "Marikina", or asks to set the location."""
    location = resolve_location(question)
    if location is None:
        return AgentReply(text=ASK_FOR_LOCATION)
    return AgentReply(
        text=f"Got it: {location.name}. Ask me for the nearest evacuation center, the flood risk, or a safe route.\n"
        f"Naitala ko: {location.name}. Itanong ang pinakamalapit na evacuation center, flood risk, o ligtas na ruta.",
        location=location,
    )
