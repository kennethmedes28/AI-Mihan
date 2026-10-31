"""Duration agent: answers "how long will the flood last?" as far as the data allows.

Nobody can predict how long floodwater will stay: it depends on the rain that
is still coming, how well the drains work, and the tide. So this agent never
gives a number of hours for the flood. It says that plainly, and then gives
what can be read from data:

    * how much rain is forecast in the next 24 hours, and when it is expected to ease
    * whether a high tide is forecast (floodwater drains slowly then)

It replaces the old behaviour of answering with the forecast risk level, which
read as "no flood" to someone already standing in water. No chat model is used.

Example:
    reply = answer_duration_question("Gaano katagal ang baha sa Marikina?", context)
"""

import logging
from datetime import datetime

from agents.backup_sources import HIGH_TIDE_M, fetch_peak_tide
from agents.fetch_supabase_open_meteo_datasets import NCR_LOCATIONS
from agents.flood_risk_agent import download_rain, split_rain
from agents.flood_types import AgentReply, ChatContext
from agents.location_agent import ASK_FOR_LOCATION, find_city
from src.rag.models import PHILIPPINE_TZ
from utils.tracing import traced_agent

logger = logging.getLogger(__name__)

DURATION = "duration"  # same value as langgraph_orchestrator.intents.DURATION

WET_HOUR_MM = 0.5  # an hour with at least this much rain counts as "still raining"
LITTLE_RAIN_TOTAL_MM = 2.0  # less rain than this in 24 hours is "no significant rain"

CANNOT_PREDICT = (
    "I cannot predict how long floodwater will stay. It depends on how much more rain falls, "
    "how well the drains work, and the tide. Ask your barangay or local DRRMO for updates.\n"
    "Hindi ko matatantiya kung gaano katagal ang baha. Depende ito sa dagdag na ulan, sa daloy ng kanal, at sa tubig-dagat. "
    "Magtanong sa barangay o lokal na DRRMO para sa updates."
)

SAFETY_REMINDER = (
    "If water is rising where you are, move to higher ground and call 911. Do not walk or drive through floodwater.\n"
    "Kung tumataas ang tubig, lumipat sa mas mataas na lugar at tumawag sa 911. Huwag tumawid o magmaneho sa baha."
)


def describe_rain(next_hours: list[float]) -> str:
    """Two lines (English, Filipino) about the rain forecast for the next 24 hours."""
    total, peak = sum(next_hours), max(next_hours)
    if total < LITTLE_RAIN_TOTAL_MM:
        return (
            "- No significant rain is forecast in the next 24 hours.\n"
            "- Walang makabuluhang ulan na inaasahan sa susunod na 24 oras."
        )

    wet_hours = [number for number, amount in enumerate(next_hours, start=1) if amount >= WET_HOUR_MM]
    if wet_hours and wet_hours[-1] >= len(next_hours):
        ease_english = "rain is still forecast at the end of the next 24 hours"
        ease_filipino = "inaasahan pa rin ang ulan sa pagtatapos ng susunod na 24 oras"
    else:
        hours = wet_hours[-1] if wet_hours else 1
        ease_english = f"the rain is forecast to ease in about {hours} hour(s)"
        ease_filipino = f"inaasahang hihina ang ulan sa loob ng mga {hours} oras"
    return (
        f"- Rain forecast: {total:.0f} mm in the next 24 hours (peak {peak:.1f} mm in one hour); {ease_english}. "
        "While it keeps raining, floodwater is unlikely to go down.\n"
        f"- Inaasahang ulan: {total:.0f} mm sa susunod na 24 oras (pinakamalakas {peak:.1f} mm kada oras); {ease_filipino}. "
        "Habang umuulan, malamang na hindi bababa ang tubig."
    )


def describe_tide(peak_tide_m: float | None) -> str:
    """A line about high tide, or nothing when there is no tide data (inland cities often have none)."""
    if peak_tide_m is None or peak_tide_m < HIGH_TIDE_M:
        return ""
    return (
        f"\n- High tide is forecast ({peak_tide_m:.1f} m) in the next 24 hours, so floodwater drains more slowly then."
        f"\n- May high tide ({peak_tide_m:.1f} m) sa susunod na 24 oras, kaya mas mabagal bumaba ang baha."
    )


def format_duration(city: str, next_hours: list[float] | None, peak_tide_m: float | None, as_of: datetime) -> str:
    """Write the whole answer. ``next_hours`` is None when the rain forecast could not be read."""
    when = as_of.astimezone(PHILIPPINE_TZ).strftime("%b %d, %I:%M %p")
    lines = [f"How long will the flood last in {city.title()}? / Gaano katagal ang baha sa {city.title()}? (as of {when})", CANNOT_PREDICT, ""]
    if next_hours:
        lines.append(describe_rain(next_hours) + describe_tide(peak_tide_m))
    else:
        lines.append("- I could not read the rain forecast, so I cannot say if more rain is coming.\n"
                     "- Hindi ko mabasa ang forecast ng ulan, kaya hindi ko masabi kung may dagdag na ulan.")
    lines += ["", SAFETY_REMINDER]
    return "\n".join(lines)


@traced_agent("flood_duration_agent")
def answer_duration_question(question: str, context: ChatContext) -> AgentReply:
    """Answer "how long will the flood last?" for the city in the question or the user's saved city."""
    city = find_city(question) or (context.location.city if context.location else None)
    if city is None:
        return AgentReply(text=ASK_FOR_LOCATION, pending_intent=DURATION)

    now = context.now or datetime.now(PHILIPPINE_TZ)
    lat, lng = NCR_LOCATIONS[city]
    try:
        _, next_hours = split_rain(download_rain(lat, lng), now)
    except Exception as error:
        logger.warning("Rain data unavailable for %s: %s", city, error)
        next_hours = None

    return AgentReply(text=format_duration(city, next_hours, fetch_peak_tide(lat, lng, now), now))
