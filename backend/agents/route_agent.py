"""Route agent: finds a way to an evacuation center that avoids flooded areas.

Steps (no chat model, so the safety check is the same every time):

    1. collect flooded areas (community reports, plus mapped spots at high risk)
    2. ask OpenRouteService for up to 3 routes that avoid those areas
    3. check every route again against the areas ourselves, since a routing
       engine can only avoid what it is told and may return a route that still clips one
    4. keep the shortest route that passes the check

The answer never says a route is "safe". It says the route avoids the areas
known at a given time, and what to do if the road ahead is flooded.
If no route passes, it says so instead of inventing one.

Needs the ORS_API_KEY environment variable (free key at openrouteservice.org).
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from agents.evacuation_agent import center_to_location, is_in_zone, nearest_safe_center
from agents.flood_risk_agent import assess_flood_risk
from agents.flood_types import AgentReply, ChatContext, FloodZone, Location
from agents.flood_zones import collect_zones
from agents.location_agent import ASK_FOR_LOCATION, resolve_location
from src.rag.models import PHILIPPINE_TZ
from utils.geo import bearing_name, circle_polygon, haversine_m, point_to_segment_m
from utils.tracing import traced_agent

logger = logging.getLogger(__name__)

ORS_URL = "https://api.openrouteservice.org/v2/directions/{profile}/geojson"
REQUEST_TIMEOUT_SECONDS = 20.0
PROFILE_DRIVING = "driving-car"
PROFILE_WALKING = "foot-walking"
STEPS_SHOWN = 8
MAX_AVOID_ZONES = 30  # ORS limits the size of the avoid area; nearest zones matter most
WALKING_WORDS = ["walk", "lakad", "naglalakad", "foot", "maglakad"]

ORIGIN_PATTERN = re.compile(r"(?:from|mula sa|galing sa)\s+(.+?)(?=\s+(?:to|papunta|patungo|going)\b|$)", re.I)
DESTINATION_PATTERN = re.compile(r"(?:\bto|papunta sa|papuntang|patungo sa|going to)\s+(.+?)(?=\s+(?:from|mula|galing)\b|$)", re.I)

ROUTING_UNAVAILABLE = (
    "I cannot calculate a route right now. Head toward {name} ({distance:.1f} km {direction}), "
    "use main roads on higher ground, and ask your barangay or call 911.\n"
    "Hindi ako makabuo ng ruta ngayon. Pumunta patungong {name} ({distance:.1f} km sa {direction}), "
    "gumamit ng pangunahing kalsada sa mataas na lugar, at magtanong sa barangay o tumawag sa 911."
)

NO_SAFE_ROUTE = (
    "I could not find a route to {name} that avoids the flooded areas I know about. "
    "Do not force a flooded road. Stay on high ground, and call 911 or your local DRRMO for rescue.\n"
    "Walang nakitang ruta papuntang {name} na umiiwas sa mga kilalang baha. "
    "Huwag dumaan sa binahang kalsada. Manatili sa mataas na lugar at tumawag sa 911 o lokal na DRRMO."
)


@dataclass
class Route:
    """One route from the routing engine."""

    distance_m: float
    duration_s: float
    coordinates: list[list[float]]  # GeoJSON order: [lng, lat]
    steps: list[str] = field(default_factory=list)


def route_hits_zone(route: Route, zone: FloodZone) -> bool:
    """True if the route's line passes through the flooded circle."""
    points = route.coordinates
    return any(
        point_to_segment_m(zone.lat, zone.lng, a[1], a[0], b[1], b[0]) <= zone.radius_m
        for a, b in zip(points, points[1:])
    )


def pick_safe_route(routes: list[Route], zones: list[FloodZone]) -> Route | None:
    """The shortest route (by time) that touches none of the zones, or None."""
    clear = [route for route in routes if not any(route_hits_zone(route, zone) for zone in zones)]
    return min(clear, key=lambda route: route.duration_s) if clear else None


def zones_to_avoid(start: Location, end: Location, zones: list[FloodZone]) -> list[FloodZone]:
    """Zones for the engine to avoid: not the ones the trip starts in, nearest to the trip first.

    A zone containing the start point cannot be avoided (the user is already in it),
    so it is left out here and the final check ignores it too.
    """
    outside_start = [zone for zone in zones if haversine_m(start.lat, start.lng, zone.lat, zone.lng) > zone.radius_m]
    outside_start.sort(key=lambda zone: min(
        haversine_m(start.lat, start.lng, zone.lat, zone.lng), haversine_m(end.lat, end.lng, zone.lat, zone.lng)
    ))
    return outside_start[:MAX_AVOID_ZONES]


def request_routes(
    start: Location,
    end: Location,
    profile: str,
    zones: list[FloodZone],
    api_key: str,
    client: httpx.Client | None = None,
) -> list[Route]:
    """Ask OpenRouteService for up to 3 routes that avoid the zones. Raises on failure."""
    body: dict = {
        "coordinates": [[start.lng, start.lat], [end.lng, end.lat]],
        "instructions": True,
        "alternative_routes": {"target_count": 3, "share_factor": 0.6, "weight_factor": 1.6},
    }
    if zones:
        body["options"] = {
            "avoid_polygons": {
                "type": "MultiPolygon",
                "coordinates": [[circle_polygon(z.lat, z.lng, z.radius_m)] for z in zones],
            }
        }

    headers = {"Authorization": api_key}
    url = ORS_URL.format(profile=profile)
    owns_client = client is None
    client = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        response = client.post(url, json=body, headers=headers)
        if response.status_code >= 400 and "alternative_routes" in body:
            # Alternatives are not allowed for some trips (for example very long ones): retry with one route.
            del body["alternative_routes"]
            response = client.post(url, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
    finally:
        if owns_client:
            client.close()

    routes = []
    for feature in data.get("features", []):
        summary = feature["properties"]["summary"]
        steps = [
            step["instruction"]
            for segment in feature["properties"].get("segments", [])
            for step in segment.get("steps", [])
        ]
        routes.append(Route(summary["distance"], summary["duration"], feature["geometry"]["coordinates"], steps))
    return routes


def parse_trip(question: str) -> tuple[str | None, str | None]:
    """Pull the "from ..." and "to ..." phrases out of a question, if present."""
    origin = ORIGIN_PATTERN.search(question)
    destination = DESTINATION_PATTERN.search(question)
    return (origin.group(1) if origin else None, destination.group(1) if destination else None)


def format_route(route: Route, destination: Location, zones: list[FloodZone], profile: str, as_of) -> str:
    """Write the route as a short bilingual message with its limits stated."""
    mode = "walking" if profile == PROFILE_WALKING else "driving"
    lines = [
        f"Route to {destination.name} ({mode}): {route.distance_m / 1000:.1f} km, about {route.duration_s / 60:.0f} min."
    ]
    lines += [f"{number}. {step}" for number, step in enumerate(route.steps[:STEPS_SHOWN], start=1)]
    if len(route.steps) > STEPS_SHOWN:
        lines.append(f"... and {len(route.steps) - STEPS_SHOWN} more steps.")

    time_text = as_of.astimezone(PHILIPPINE_TZ).strftime("%I:%M %p")
    if zones:
        reports = sum(zone.source == "report" for zone in zones)
        lines.append(
            f"This route avoids {len(zones)} flooded area(s) known at {time_text} "
            f"({reports} community report(s), {len(zones) - reports} mapped spot(s)). It is not a guarantee: "
            "water can rise quickly and many streets are not reported."
        )
    else:
        lines.append(
            f"No flooded areas were reported or mapped at {time_text}, so I could not check this route against floods."
        )
    lines.append(
        "If the road ahead is flooded, turn back. Do not walk or drive through moving water. "
        "Kung binaha ang daan, bumalik. Huwag tumawid sa umaagos na tubig."
    )
    return "\n".join(lines)


@traced_agent("route_agent")
def answer_route_question(question: str, context: ChatContext) -> AgentReply:
    """Answer "how do I get there?" with a route that avoids known flooded areas."""
    origin_text, destination_text = parse_trip(question)
    if origin_text:
        start = resolve_location(origin_text)
    elif destination_text:
        start = None  # any place in the question is the destination, not the start
    else:
        start = resolve_location(question)  # "I am in Pasig, how do I get out?"
    start = start or context.location
    if start is None:
        return AgentReply(text=ASK_FOR_LOCATION, pending_intent="route")

    risk = assess_flood_risk(start.city, now=context.now)
    zones = collect_zones(context, start.city, risk.level)

    destination = (resolve_location(destination_text) if destination_text else None) or context.destination
    if destination is None:
        center = nearest_safe_center(start, context, zones)
        if center is None:
            return AgentReply(
                text="Tell me where you want to go, or ask for the nearest evacuation center first.\n"
                "Sabihin kung saan kayo pupunta, o itanong muna ang pinakamalapit na evacuation center.",
                location=start,
            )
        destination = center_to_location(center)

    if is_in_zone(destination.lat, destination.lng, zones):
        return AgentReply(
            text=f"{destination.name} is inside a flooded area. Ask for the nearest evacuation center to find another.\n"
            f"Nasa binahang lugar ang {destination.name}. Itanong ang pinakamalapit na evacuation center para sa iba.",
            location=start,
        )

    api_key = context.extra.get("ors_api_key") or os.getenv("ORS_API_KEY")
    distance_km = haversine_m(start.lat, start.lng, destination.lat, destination.lng) / 1000
    fallback = ROUTING_UNAVAILABLE.format(
        name=destination.name,
        distance=distance_km,
        direction=bearing_name(start.lat, start.lng, destination.lat, destination.lng),
    )
    if not api_key:
        logger.warning("ORS_API_KEY is not set; routing is unavailable.")
        return AgentReply(text=fallback, location=start, destination=destination)

    profile = PROFILE_WALKING if any(word in question.lower() for word in WALKING_WORDS) else PROFILE_DRIVING
    avoid = zones_to_avoid(start, destination, zones)
    try:
        routes = request_routes(start, destination, profile, avoid, api_key)
    except Exception as error:
        logger.error("Routing failed: %s", error)
        return AgentReply(text=fallback, location=start, destination=destination)

    best = pick_safe_route(routes, avoid)
    if best is None:
        return AgentReply(text=NO_SAFE_ROUTE.format(name=destination.name), location=start, destination=destination)

    as_of = context.now or datetime.now(PHILIPPINE_TZ)
    return AgentReply(
        text=format_route(best, destination, avoid, profile, as_of), location=start, destination=destination
    )
