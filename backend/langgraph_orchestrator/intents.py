"""Decide what a user's message is about, using keywords only.

No model is called here, so routing costs nothing. The order of the checks
matters: an emergency always wins, then the specific flood services (report,
route, evacuation, contacts), then preparedness, then flood risk and weather.
Anything unrecognised goes to the document agent, which answers from the
stored flood material or says it has no information.

A message can ask for several things at once ("forecast for 7 days, what do I
do if the flood is knee-deep, and where is the nearest evacuation center?").
``split_by_intent`` cuts such a message into one question per topic. A request to
report a flood is one of those topics, so it is saved and the rest is still answered.
"""

import re

EMERGENCY = "emergency"
REPORT = "report"
ROUTE = "route"
EVACUATION = "evacuation"
CONTACTS = "contacts"
DURATION = "duration"
PREPAREDNESS = "preparedness"
FLOOD_RISK = "flood_risk"
WEATHER = "weather"
GENERAL = "general"
MULTI = "multi"  # a message with several topics; see split_by_intent

# Checked in this order; the first list with a match decides the intent.
INTENT_KEYWORDS = [
    (
        EMERGENCY,
        [
            "emergency", "trapped", "stranded", "rescue", "saklolo", "nakulong", "naiipit", "nalulunod", "drowning",
            # Water deep enough to knock a person over (waist or higher), or people already on a roof.
            # Knee-deep ("hanggang tuhod") is not here: that user still needs to be told where to go.
            "hanggang baywang", "hanggang bewang", "hanggang dibdib", "hanggang leeg",
            "waist deep", "waist-deep", "chest deep", "chest-deep", "neck deep", "neck-deep",
            "nasa bubong", "sa bubong", "rooftop",
            "swept away", "inanod", "inaanod", "hindi makalabas", "di makalabas", "can't get out", "cannot get out",
            "lampas tao", "lampas-tao", "over my head",
        ],
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
        DURATION,
        # "How long will the flood last?" is not a forecast question: it needs its own answer (and its own limits).
        [
            "how long", "gaano katagal", "magtatagal", "tatagal", "hanggang kailan", "until when",
            "kailan titila", "kailan bababa", "kailan mawawala", "how many hours", "ilang oras",
        ],
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


# Depth phrases that mean water has reached a person's thighs or higher. Thigh-deep water already
# knocks people over, so these are emergencies even when the exact wording is not in the list above.
# Knee-deep is not here: that user still gets the centers and safety steps from the evacuation agent.
EMERGENCY_PATTERNS = [
    re.compile(r"hanggang\s+(?:sa\s+)?(?:hita|baywang|bewang|dibdib|leeg|balikat|tiyan|ulo)"),
    re.compile(r"up\s+to\s+(?:my\s+|the\s+|our\s+)?(?:waist|chest|neck|thighs?|shoulders?)"),
    re.compile(r"(?:waist|chest|neck|thigh)[\s-]*(?:deep|high)"),
    re.compile(r"(?:above|over)\s+(?:my\s+|the\s+)?(?:waist|chest)"),
]


def classify_intent(message: str, skip_emergency: bool = False) -> str:
    """Return the intent name, or ``general`` when nothing matches.

    ``skip_emergency`` is for one part of a longer message: the whole message is already an
    emergency, and this part ("I want to report a flood up to my waist") still has its own topic.
    """
    text = message.lower()
    for intent, keywords in INTENT_KEYWORDS:
        if intent == EMERGENCY:
            if skip_emergency:
                continue
            if any(keyword in text for keyword in keywords) or any(p.search(text) for p in EMERGENCY_PATTERNS):
                return EMERGENCY
        elif any(keyword in text for keyword in keywords):
            return intent
    return GENERAL


# --- Messages with several topics ---------------------------------------------

# Topics that get their own answer when a message mixes them, in answer order. A report comes
# first: it is an action, and it must not be lost because the user also asked something else.
# Emergency is not here: it is handled before any topic (see the emergency node in graph.py).
MULTI_TOPIC_ORDER = [REPORT, WEATHER, FLOOD_RISK, PREPAREDNESS, EVACUATION, ROUTE, DURATION, CONTACTS]

# Flood risk is a generic topic ("baha" appears in almost every question). It only counts as its own
# topic when none of these more specific ones is present, so "what do I do if it floods?" stays one topic.
SPECIFIC_FLOOD_TOPICS = [PREPAREDNESS, EVACUATION, ROUTE, DURATION, CONTACTS]

# A message is cut into parts at sentence marks, commas, and the words "at" / "and" / "tapos" / "then".
CLAUSE_BREAK = re.compile(r"[.?!;,]|\s+(?:at|and|tapos|then)\s+", re.IGNORECASE)


def split_by_intent(message: str, min_topics: int = 2, in_emergency: bool = False) -> list[tuple[str, str]]:
    """Split a message into ``(intent, question)`` pairs, one per topic, in answer order.

    Returns an empty list unless the message really has ``min_topics`` or more topics, so a
    normal single-topic message keeps going through ``classify_intent``. An emergency message
    is never split, unless the emergency node asks for it with ``in_emergency=True`` (it
    answers the emergency first and then the topics, so it passes ``min_topics=1``).

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
    if classify_intent(message) == EMERGENCY and not in_emergency:
        return []

    clauses = [clause.strip() for clause in CLAUSE_BREAK.split(message) if clause and clause.strip()]
    clause_intents = [classify_intent(clause, skip_emergency=True) for clause in clauses]

    topics = [topic for topic in MULTI_TOPIC_ORDER if topic in clause_intents]
    if FLOOD_RISK in topics and any(topic in topics for topic in SPECIFIC_FLOOD_TOPICS):
        topics.remove(FLOOD_RISK)
    if len(topics) < min_topics:
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
