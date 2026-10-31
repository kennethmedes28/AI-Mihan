"""Coverage agent: the reply for a place outside Metro Manila.

The project only has flood data, evacuation centers and hotlines for Metro Manila
(NCR). For a place such as Tanza, Cavite the other agents would only ask "which
city?" again, which tells the user nothing. This agent says plainly that the area
is not covered, gives the safety steps that are true anywhere, and lists the
national hotlines (copied from the contacts file, never written by a model).

Example:
    reply = answer_outside_coverage("Nasa Tanza, Cavite ako. May data ka ba ng baha?", ChatContext())
"""

from agents.contacts_agent import contacts_for_city, load_contacts
from agents.flood_types import AgentReply, ChatContext
from agents.location_agent import find_outside_place
from utils.tracing import traced_agent

# MMDA only works in Metro Manila, so it is not listed for other places.
NATIONAL_ONLY_EXCLUDED_AGENCIES = {"MMDA"}

OUTSIDE_COVERAGE_ANSWER = (
    "I only have flood data, evacuation centers and hotlines for Metro Manila (NCR), so I cannot answer for {place}.\n"
    "For {place}: if you are in danger, call 911. Contact your city or provincial DRRMO, "
    "and check pagasa.dost.gov.ph and ndrrmc.gov.ph. Move to higher ground if water is rising.\n"
    "Metro Manila (NCR) lang ang saklaw ng aking datos ng baha, evacuation center at hotline, "
    "kaya hindi ako makasasagot para sa {place}.\n"
    "Para sa {place}: kung nasa panganib, tumawag sa 911. Makipag-ugnayan sa DRRMO ng inyong lungsod o probinsya, "
    "at tingnan ang pagasa.dost.gov.ph at ndrrmc.gov.ph. Lumipat sa mas mataas na lugar kung tumataas ang tubig."
)


def format_national_numbers() -> str:
    """The national hotlines as a short list (no city-specific lines)."""
    contacts = [row for row in load_contacts() if row["Agency"] not in NATIONAL_ONLY_EXCLUDED_AGENCIES]
    lines = [f"- {agency}: {', '.join(numbers)}" for agency, numbers in contacts_for_city(None, contacts)]
    return "National emergency numbers:\n" + "\n".join(lines) if lines else ""


def outside_coverage_text(place: str) -> str:
    """The "not covered" message for a place, with the national numbers."""
    text = OUTSIDE_COVERAGE_ANSWER.format(place=place)
    numbers = format_national_numbers()
    return f"{text}\n\n{numbers}" if numbers else text


@traced_agent("coverage_agent")
def answer_outside_coverage(question: str, context: ChatContext) -> AgentReply:
    """Say that the place is outside Metro Manila, and give the safety steps and national numbers."""
    place = find_outside_place(question) or "that area"
    return AgentReply(text=outside_coverage_text(place))
