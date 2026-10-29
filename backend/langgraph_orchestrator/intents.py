"""Decide what a user's message is about, using keywords only.

No model is called here, so routing costs nothing. The order of the checks
matters: an emergency always wins, then the specific flood services (report,
route, evacuation, contacts), then preparedness, then flood risk and weather.
Anything unrecognised goes to the document agent, which answers from the
stored flood material or says it has no information.

A message can ask for several things at once ("forecast for 7 days, what do I
do if the flood is knee-deep, and where is the nearest evacuation center?").
``split_by_intent`` cuts such a message into one question per topic.
"""

import re

EMERGENCY = "emergency"
REPORT = "report"
ROUTE = "route"
EVACUATION = "evacuation"
CONTACTS = "contacts"
PREPAREDNESS = "preparedness"
FLOOD_RISK = "flood_risk"
WEATHER = "weather"
GENERAL = "general"
MULTI = "multi"  # a message with several topics; see split_by_intent

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
        [
            "prepar", "handa", "checklist", "go bag", "evacuate", "lumikas", "what to do", "ano ang gagawin",
            "dapat gawin", "dapat kong gawin", "kailangan kong gawin", "kailangan gawin", "what should i do",
        ],
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


# --- Messages with several topics ---------------------------------------------

# Topics that get their own answer when a message mixes them. Emergency and report
# are not here: an emergency always answers alone, and a report is an action.
MULTI_TOPIC_ORDER = [WEATHER, FLOOD_RISK, PREPAREDNESS, EVACUATION, ROUTE, CONTACTS]

# Flood risk is a generic topic ("baha" appears in almost every question). It only counts as its own
# topic when none of these more specific ones is present, so "what do I do if it floods?" stays one topic.
SPECIFIC_FLOOD_TOPICS = [PREPAREDNESS, EVACUATION, ROUTE, CONTACTS]

# A message is cut into parts at sentence marks, commas, and the words "at" / "and" / "tapos" / "then".
CLAUSE_BREAK = re.compile(r"[.?!;,]|\s+(?:at|and|tapos|then)\s+", re.IGNORECASE)


def split_by_intent(message: str) -> list[tuple[str, str]]:
    """Split a message into ``(intent, question)`` pairs, one per topic, in answer order.

    Returns an empty list unless the message really has two or more topics, so a
    normal single-topic message keeps going through ``classify_intent``.

    Each part is cut at sentence marks, commas, "at" and "and":

        * a part with a topic keyword belongs to that topic;
        * a part with no topic ("Malapit kami sa dagat") is context, added to every question;
        * a flood part that lost to a more specific topic ("kung bumaha hanggang tuhod")
          is also context, but only for the non-weather questions.

    Example:
        "Mag forecast ka ng 7 days at kung bumaha hanggang tuhod, ano ang gagawin ko at saan ang evacuation center?"
        -> [(WEATHER, "Mag forecast ka ng 7 days"),
            (PREPAREDNESS, "kung bumaha hanggang tuhod ano ang gagawin ko"),
            (EVACUATION, "kung bumaha hanggang tuhod saan ang evacuation center")]
    """
    if classify_intent(message) == EMERGENCY:
        return []

    clauses = [clause.strip() for clause in CLAUSE_BREAK.split(message) if clause and clause.strip()]
    clause_intents = [classify_intent(clause) for clause in clauses]

    topics = [topic for topic in MULTI_TOPIC_ORDER if topic in clause_intents]
    if FLOOD_RISK in topics and any(topic in topics for topic in SPECIFIC_FLOOD_TOPICS):
        topics.remove(FLOOD_RISK)
    if len(topics) < 2:
        return []

    context, flood_context = [], []
    questions = {topic: [] for topic in topics}
    for clause, intent in zip(clauses, clause_intents):
        if intent in questions:
            questions[intent].append(clause)
        elif intent == FLOOD_RISK:
            flood_context.append(clause)
        else:
            context.append(clause)

    parts = []
    for topic in topics:
        extra = [] if topic == WEATHER else flood_context
        parts.append((topic, " ".join(questions[topic] + extra + context)))
    return parts
