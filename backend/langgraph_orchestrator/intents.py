"""Decide what a user's message is about, using keywords only.

No model is called here, so routing costs nothing. The order of the checks
matters: an emergency always wins, then the specific flood services (report,
route, evacuation, contacts), then preparedness, then flood risk and weather.
Anything unrecognised goes to the document agent, which answers from the
stored flood material or says it has no information.
"""

EMERGENCY = "emergency"
REPORT = "report"
ROUTE = "route"
EVACUATION = "evacuation"
CONTACTS = "contacts"
PREPAREDNESS = "preparedness"
FLOOD_RISK = "flood_risk"
WEATHER = "weather"
GENERAL = "general"

# Checked in this order; the first list with a match decides the intent.
INTENT_KEYWORDS = [
    (
        EMERGENCY,
        ["emergency", "trapped", "stranded", "rescue", "saklolo", "nakulong", "naiipit", "nalulunod", "drowning"],
    ),
    (
        REPORT,
        ["report", "i-report", "ireport", "iulat", "isumbong", "magreport", "mag-report", "nag-report"],
    ),
    (
        ROUTE,
        [
            "route", "ruta", "direction", "paano pumunta", "paano makarating", "how do i get", "how to get",
            "safe way", "ligtas na daan", "daan papunta", "shortcut", "makakadaan", "can i pass", "madadaanan",
        ],
    ),
    (
        EVACUATION,
        [
            "evacuation", "evac center", "evacuation center", "shelter", "pinakamalapit", "nearest center",
            "saan lilikas", "saan pupunta", "where to go", "silungan", "higher ground",
        ],
    ),
    (
        CONTACTS,
        ["hotline", "contact", "phone number", "numero", "drrmo", "red cross", "tawagan", "who to call"],
    ),
    (
        PREPAREDNESS,
        # Word stems, so "prepare", "preparing", "maghanda" and "ihanda" all match.
        ["prepar", "handa", "checklist", "go bag", "evacuate", "lumikas", "what to do", "ano ang gagawin", "dapat gawin"],
    ),
    (
        FLOOD_RISK,
        [
            "flood", "baha", "babaha", "bumaha", "lubog", "risk", "river level", "antas ng ilog", "water level",
            "ligtas ba", "is it safe", "ilog", "river", "tubig",
        ],
    ),
    (
        WEATHER,
        [
            "weather", "panahon", "rain", "ulan", "uulan", "typhoon", "bagyo", "storm",
            "habagat", "wind", "hangin", "forecast", "signal", "pagasa",
        ],
    ),
]


def classify_intent(message: str) -> str:
    """Return the intent name, or ``general`` when nothing matches."""
    text = message.lower()
    for intent, keywords in INTENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return intent
    return GENERAL
