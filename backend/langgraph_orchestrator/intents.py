"""Decide what a user's message is about, using keywords only.

No model is called here, so routing costs nothing. The order of the checks
matters: an emergency always wins, then preparedness questions, then the
specific hazards. Anything unrecognised goes to the document agent, which
answers from the stored preparedness material or says it has no information.
"""

EMERGENCY = "emergency"
PREPAREDNESS = "preparedness"
EARTHQUAKE = "earthquake"
WEATHER = "weather"

# Checked in this order; the first list with a match decides the intent.
INTENT_KEYWORDS = [
    (
        EMERGENCY,
        ["emergency", "trapped", "stranded", "rescue", "saklolo", "nakulong", "naiipit", "nalulunod", "drowning"],
    ),
    (
        PREPAREDNESS,
        # Word stems, so "prepare", "preparing", "maghanda" and "ihanda" all match.
        ["prepar", "handa", "checklist", "go bag", "evacuate", "lumikas"],
    ),
    (
        EARTHQUAKE,
        ["earthquake", "lindol", "quake", "phivolcs", "magnitude", "aftershock", "tsunami"],
    ),
    (
        WEATHER,
        [
            "weather", "panahon", "rain", "ulan", "uulan", "flood", "baha", "typhoon", "bagyo", "storm",
            "habagat", "wind", "hangin", "heat", "init", "forecast", "signal", "pagasa",
        ],
    ),
]


def classify_intent(message: str) -> str:
    """Return ``emergency``, ``preparedness``, ``earthquake``, ``weather`` or ``general``."""
    text = message.lower()
    for intent, keywords in INTENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return intent
    return "general"
