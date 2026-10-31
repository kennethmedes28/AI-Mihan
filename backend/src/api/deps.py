"""Pieces the routers share: the chat graph, the database, the rate limits and the error texts.

The graph and the database connection are created once at start-up (``startup.py``) and kept
on ``app.state``. Tests put fakes there instead, so no router needs to know where they come from.
"""

import hmac
import logging
from collections.abc import Callable

from fastapi import HTTPException, Request

from agents.location_agent import find_city, nearest_city
from src.api.schemas import Coordinates

logger = logging.getLogger(__name__)

# Every error that a person in a flood might see also says what to do next.
DATABASE_UNAVAILABLE = (
    "The service cannot reach its data right now. If you are in danger, call 911 and contact your barangay or local DRRMO.\n"
    "Hindi maabot ng serbisyo ang datos ngayon. Kung nasa panganib, tumawag sa 911 at makipag-ugnayan sa barangay o lokal na DRRMO."
)
TOO_MANY_REQUESTS = (
    "Too many requests. Please wait a moment and try again. If you are in danger, call 911.\n"
    "Masyadong maraming request. Maghintay sandali at subukan muli. Kung nasa panganib, tumawag sa 911."
)
UNKNOWN_CITY = "Unknown city. Only the cities of Metro Manila (NCR) are covered, for example Manila, Quezon City or Marikina."


STAFF_KEY_REQUIRED = "A valid staff key is required. Send it as 'Authorization: Bearer <key>'."


def require_staff(request: Request) -> None:
    """Allow the call only when it carries the staff key.

    The key is compared in constant time, so the time taken does not hint at how much of a
    guess was right. The answer is the same whether the header is missing or wrong.
    """
    expected = request.app.state.settings.staff_api_key
    header = request.headers.get("authorization", "")
    scheme, _, given = header.partition(" ")

    key_matches = bool(expected) and scheme.lower() == "bearer" and hmac.compare_digest(given.strip().encode(), expected.encode())
    if not key_matches:
        raise HTTPException(status_code=401, detail=STAFF_KEY_REQUIRED, headers={"WWW-Authenticate": "Bearer"})


def get_graph(request: Request):
    """The compiled chat graph, or a 503 while the service is still starting."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE)
    return graph


def get_db(request: Request) -> Callable:
    """A function that returns the open database connection (it reconnects if the connection dropped)."""
    get_connection = getattr(request.app.state, "get_connection", None)
    if get_connection is None:
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE)
    return get_connection


def database_unavailable(error: Exception) -> HTTPException:
    """Log a database failure and return the 503 to raise.

    Only the error text is logged, never what the user sent, so reports and requests
    (which can hold names and phone numbers) do not end up in the logs.
    """
    logger.error("Database call failed: %s", error)
    return HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE)


def client_address(request: Request) -> str:
    """Who is calling, for the rate limit.

    Behind a proxy you control, the real address is the first one in ``X-Forwarded-For``.
    That header is only trusted when ``TRUST_PROXY_HEADERS`` is on, because without a proxy
    anyone can send a fake one and dodge the limit.
    """
    if request.app.state.settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def limit_requests(name: str) -> Callable:
    """Build a dependency that applies the named rate limit (``chat``, ``reports`` or ``requests``)."""

    def check(request: Request) -> None:
        wait = request.app.state.limiters[name].retry_after(client_address(request))
        if wait is not None:
            raise HTTPException(status_code=429, detail=TOO_MANY_REQUESTS, headers={"Retry-After": str(wait)})

    return check


def resolve_city(city_text: str | None, location: Coordinates | None) -> str:
    """The Metro Manila city key ("paranaque", "quezon city") for a report or request.

    A city name wins when given; otherwise the nearest city to the GPS point is used.
    """
    if city_text:
        city = find_city(city_text)
        if city is None:
            raise HTTPException(status_code=422, detail=UNKNOWN_CITY)
        return city
    return nearest_city(location.lat, location.lng)
