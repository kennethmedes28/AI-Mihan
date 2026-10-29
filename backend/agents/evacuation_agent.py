"""Evacuation agent: finds the nearest evacuation centers that are not flooded.

Steps (no chat model):

    1. read the centers of the user's city from the database
    2. drop centers that sit inside a flooded area (reported, or mapped while risk is high)
    3. sort by straight-line distance and list the closest few

The best center is remembered as the destination, so "how do I get there?"
can be answered by the route agent on the next message.
"""

import logging

from agents.flood_risk_agent import assess_flood_risk
from agents.flood_types import AgentReply, ChatContext, FloodZone, Location, PRECISION_CITY
from agents.flood_zones import collect_zones
from agents.location_agent import ASK_FOR_LOCATION, describe_precision, nearest_city, resolve_location
from src.services.fetch_postgre_database import fetch_evacuation_centers
from utils.geo import bearing_name, haversine_m
from utils.tracing import traced_agent

logger = logging.getLogger(__name__)

CENTERS_SHOWN = 3
FETCH_LIMIT = 50

NO_DATABASE_ANSWER = (
    "I cannot read the evacuation center list right now. Please ask your barangay or local DRRMO, or call 911.\n"
    "Hindi ko mabasa ang listahan ng evacuation center ngayon. Magtanong sa barangay o lokal na DRRMO, o tumawag sa 911."
)


def is_in_zone(lat: float, lng: float, zones: list[FloodZone]) -> bool:
    """True if the point lies inside any flooded area."""
    return any(haversine_m(lat, lng, zone.lat, zone.lng) <= zone.radius_m for zone in zones)


def rank_centers(origin: Location, rows: list[dict], zones: list[FloodZone]) -> tuple[list[dict], int]:
    """Sort centers by distance from the origin and drop flooded ones.

    Returns ``(centers, flooded_count)``. Each center gets a ``distance_m``.
    Centers without a position are skipped, since they cannot be placed.
    """
    usable, flooded = [], 0
    for row in rows:
        if row.get("lat") is None or row.get("lng") is None:
            continue
        if is_in_zone(row["lat"], row["lng"], zones):
            flooded += 1
            continue
        usable.append({**row, "distance_m": haversine_m(origin.lat, origin.lng, row["lat"], row["lng"])})
    usable.sort(key=lambda row: row["distance_m"])
    return usable, flooded


def center_to_location(center: dict) -> Location:
    """Turn a database row into a destination location."""
    return Location(
        name=center["name"],
        lat=center["lat"],
        lng=center["lng"],
        city=(center.get("city") or nearest_city(center["lat"], center["lng"])).lower(),
        barangay=center.get("barangay"),
        precision="gps",
    )


def nearest_safe_center(origin: Location, context: ChatContext, zones: list[FloodZone]) -> dict | None:
    """The closest center not inside a flooded area, or None. Used by the route agent too."""
    if context.get_connection is None:
        return None
    try:
        rows = fetch_evacuation_centers(context.get_connection(), city=origin.city, limit=FETCH_LIMIT)
        if not rows:  # a city with no listed centers: look across Metro Manila
            rows = fetch_evacuation_centers(context.get_connection(), limit=FETCH_LIMIT)
    except Exception as error:
        logger.error("Could not read evacuation centers: %s", error)
        return None
    centers, _ = rank_centers(origin, rows, zones)
    return centers[0] if centers else None


@traced_agent("evacuation_agent")
def answer_evacuation_question(question: str, context: ChatContext) -> AgentReply:
    """List the nearest safe evacuation centers for the user's location."""
    origin = resolve_location(question) or context.location
    if origin is None:
        return AgentReply(text=ASK_FOR_LOCATION, pending_intent="evacuation")
    if context.get_connection is None:
        return AgentReply(text=NO_DATABASE_ANSWER, location=origin)

    try:
        rows = fetch_evacuation_centers(context.get_connection(), city=origin.city, limit=FETCH_LIMIT)
        searched_everywhere = False
        if not rows:
            rows = fetch_evacuation_centers(context.get_connection(), limit=FETCH_LIMIT)
            searched_everywhere = True
    except Exception as error:
        logger.error("Could not read evacuation centers: %s", error)
        return AgentReply(text=NO_DATABASE_ANSWER, location=origin)

    risk = assess_flood_risk(origin.city, now=context.now)
    zones = collect_zones(context, origin.city, risk.level)
    centers, flooded_count = rank_centers(origin, rows, zones)

    if not centers:
        return AgentReply(
            text="I found no evacuation center I can place on the map. Call your local DRRMO or 911.\n"
            "Wala akong nakitang evacuation center na maitatala sa mapa. Tumawag sa lokal na DRRMO o 911.",
            location=origin,
        )

    lines = [f"Nearest evacuation centers to {origin.name} (flood risk: {risk.level.upper()}):"]
    for number, center in enumerate(centers[:CENTERS_SHOWN], start=1):
        direction = bearing_name(origin.lat, origin.lng, center["lat"], center["lng"])
        capacity = f", capacity {center['capacity']}" if center.get("capacity") else ""
        kind = f", {center['type'].lower()}" if center.get("type") else ""
        lines.append(
            f"{number}. {center['name']} ({center.get('barangay') or center.get('city')}{kind}) - "
            f"{center['distance_m'] / 1000:.1f} km {direction}{capacity}"
        )
    if flooded_count:
        lines.append(f"{flooded_count} center(s) skipped because they are in reported or known flooded areas.")
    if searched_everywhere:
        lines.append("No centers are listed for your city, so these are from across Metro Manila.")
    if origin.precision == PRECISION_CITY:
        lines.append(describe_precision(origin))
    lines.append(
        "Distances are in a straight line and map positions are approximate. Confirm with your barangay that the center is open.\n"
        "Say \"how do I get there?\" for a route that avoids flooded areas.\n"
        "Diretsong linya ang layo at tinatayang posisyon lamang. Kumpirmahin sa barangay kung bukas ang center. "
        "Sabihin ang \"paano pumunta\" para sa ruta na umiiwas sa baha."
    )
    return AgentReply(
        text="\n".join(lines),
        location=origin,
        destination=center_to_location(centers[0]),
    )
