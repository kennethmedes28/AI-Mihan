"""Report agent: lets residents report a flooded street, so others can avoid it.

Reports are saved as ``unverified`` with ``community`` trust. They never
override PAGASA or NDRRMC material, but the route agent avoids the reported
spot for a few hours (see ``flood_zones.py``).

A report needs a place. Exact spots work best: share a pin or coordinates, or
name the barangay. A city alone is saved too, but it cannot be drawn on the map.
"""

import logging

from agents.flood_types import PRECISION_CITY, AgentReply, ChatContext
from agents.location_agent import ASK_FOR_LOCATION, resolve_location
from src.services.fetch_postgre_database import insert_report
from utils.tracing import traced_agent

logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 500

NO_DATABASE_ANSWER = (
    "I cannot save reports right now. Please tell your barangay or local DRRMO.\n"
    "Hindi ako makapag-save ng ulat ngayon. Sabihin sa barangay o lokal na DRRMO."
)


@traced_agent("report_agent")
def answer_report_request(question: str, context: ChatContext) -> AgentReply:
    """Save the user's flood report using the place in the message or their saved location."""
    place = resolve_location(question) or context.location
    if place is None:
        return AgentReply(text=ASK_FOR_LOCATION, pending_intent="report")
    if context.get_connection is None:
        return AgentReply(text=NO_DATABASE_ANSWER, location=place)

    on_map = place.precision != PRECISION_CITY  # a city centre would mark the wrong street
    try:
        insert_report(
            context.get_connection(),
            hazard="flood",
            city=place.city,
            barangay=place.barangay,
            body=question.strip()[:MAX_BODY_CHARS],
            lat=place.lat if on_map else None,
            lng=place.lng if on_map else None,
        )
    except Exception as error:
        logger.error("Could not save report: %s", error)
        return AgentReply(text=NO_DATABASE_ANSWER, location=place)

    note = (
        "Routes will avoid this spot for the next few hours."
        if on_map
        else "I only know the city, so routes cannot avoid it yet. Send your barangay or pin to mark the street."
    )
    return AgentReply(
        text=f"Thank you. Your flood report for {place.name} was saved as an unverified community report. {note}\n"
        f"Salamat. Naitala ang ulat ng baha sa {place.name} bilang hindi pa na-verify na ulat ng komunidad.\n"
        "If someone is in danger, call 911.",
        location=place,
    )
