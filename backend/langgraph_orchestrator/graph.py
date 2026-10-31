"""The chat graph: routes each message to the right flood agent.

    START --> (route by keywords) --+--> emergency    fixed reply + local hotlines, then the rest of the message
                                    |                 (save the report, list centers, answer "how long"); no model call
                                    +--> report       save a community flood report
                                    +--> route        route to an evacuation center that avoids floods
                                    +--> evacuation   nearest evacuation centers that are not flooded
                                    +--> contacts     hotlines for the user's city
                                    +--> duration     "how long will the flood last?": rain and tide, and what cannot be known
                                    +--> outside      a place outside Metro Manila: not covered, national numbers
                                    +--> location    a place name alone ("Marikina") sets where the user is
                                    +--> flood_risk   rule-based flood risk for a city
                                    +--> weather      PAGASA + Open-Meteo
                                    +--> documents    stored flood material (RAG)
                                    +--> flood_media  a photo or video was attached: flood water (and rough depth) in it
                                    +--> multi        one message with several topics: each topic is answered by its own
                                    |                 agent (at the same time) and the answers are joined under headings
                                    |
                                    +--> END

Every node adds one assistant message to the conversation. The graph also
remembers, per conversation:

    * ``location``       where the user is
    * ``destination``    the evacuation center last suggested, for "how do I get there?"
    * ``pending_intent`` set when an agent asked "which city are you in?", so the
                         next message (just "Marikina") goes back to that agent
    * ``media``          a photo or video to check, for ONE message only (see ``media_from_dict``)
    * ``pending_parts``  the topics of a multi-topic message that are still waiting for the user's
                         location (the answer to "which city?" completes them)

When a checkpointer is given, the conversation is saved per ``thread_id``
(see checkpointer/postgres_checkpointer.py).

Example:
    graph = build_graph(llm, retriever, checkpointer, get_connection=get_connection)
    reply = await ask(graph, "Saan ang pinakamalapit na evacuation center?", "user-123",
                      location={"lat": 14.65, "lng": 121.10})
    reply = await ask(graph, "Baha dito", "user-123",
                      media={"kind": "image", "path": "/uploads/street.jpg", "estimate_depth": True})
"""

import contextvars
import ipaddress
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Callable, TypedDict
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.contacts_agent import answer_contacts_question, format_contacts
from agents.coverage_agent import answer_outside_coverage, outside_coverage_text
from agents.document_agents import answer_document_question
from agents.evacuation_agent import answer_evacuation_question
from agents.flood_media_agent import answer_flood_media_question
from agents.flood_duration_agent import answer_duration_question
from agents.flood_risk_agent import answer_flood_risk_question
from agents.flood_types import AgentReply, ChatContext, Location
from agents.location_agent import (
    ASK_FOR_LOCATION,
    answer_location_message,
    find_outside_place,
    location_from_coordinates,
    mentions_place,
    resolve_location,
)
from agents.report_agent import answer_report_request
from agents.safe_answer import safe_reply
from agents.route_agent import answer_route_question
from agents.weather_agent import answer_weather_question
from langgraph_orchestrator.intents import (
    CONTACTS,
    DURATION,
    EMERGENCY,
    EVACUATION,
    FLOOD_RISK,
    GENERAL,
    MULTI,
    PREPAREDNESS,
    REPORT,
    ROUTE,
    WEATHER,
    classify_intent,
    split_by_intent,
)
from src.rag.retriever import HybridRetriever
from utils.input_guard import blocked_reply, check_message
from video.sam_client import ImageInput, VideoInput

logger = logging.getLogger(__name__)

# Node names.
EMERGENCY_NODE = "emergency"
REPORT_NODE = "report"
ROUTE_NODE = "route"
EVACUATION_NODE = "evacuation"
CONTACTS_NODE = "contacts"
FLOOD_RISK_NODE = "flood_risk"
WEATHER_NODE = "weather"
DOCUMENTS_NODE = "documents"
LOCATION_NODE = "location"
FLOOD_MEDIA_NODE = "flood_media"
MULTI_NODE = "multi"
DURATION_NODE = "duration"
OUTSIDE_NODE = "outside"
ALL_NODES = [
    EMERGENCY_NODE, REPORT_NODE, ROUTE_NODE, EVACUATION_NODE,
    CONTACTS_NODE, FLOOD_RISK_NODE, WEATHER_NODE, DOCUMENTS_NODE, LOCATION_NODE, FLOOD_MEDIA_NODE, MULTI_NODE,
    DURATION_NODE, OUTSIDE_NODE,
]

# Which node answers each intent. Preparedness and anything unrecognised go to the documents.
NODE_FOR_INTENT = {
    EMERGENCY: EMERGENCY_NODE,
    REPORT: REPORT_NODE,
    ROUTE: ROUTE_NODE,
    EVACUATION: EVACUATION_NODE,
    CONTACTS: CONTACTS_NODE,
    FLOOD_RISK: FLOOD_RISK_NODE,
    DURATION: DURATION_NODE,
    WEATHER: WEATHER_NODE,
    PREPAREDNESS: DOCUMENTS_NODE,
    MULTI: MULTI_NODE,  # only reached as a waiting question; new multi-topic messages are detected in route_message
}

# Topics whose agents need to know where the user is.
TOPICS_NEEDING_LOCATION = {REPORT, FLOOD_RISK, EVACUATION, ROUTE, DURATION, CONTACTS}

# These topics have no data outside Metro Manila. Weather and preparedness do not depend on the city,
# so a message about them is still answered even when it names a place outside Metro Manila.
TOPICS_OUTSIDE_COVERAGE_STILL_ANSWERED = {WEATHER, PREPAREDNESS}

# Heading above each topic's answer in a combined reply.
TOPIC_HEADINGS = {
    REPORT: "Flood report / Ulat ng baha",
    WEATHER: "Weather forecast / Lagay ng panahon",
    FLOOD_RISK: "Flood risk / Panganib ng baha",
    PREPAREDNESS: "What to do / Ano ang gagawin",
    EVACUATION: "Evacuation center",
    ROUTE: "Route / Ruta",
    DURATION: "How long / Gaano katagal",
    CONTACTS: "Hotlines",
}

# Fixed reply for emergencies: instant, free, and does not depend on any API.
# 911 is the Philippines' national emergency number.
EMERGENCY_ANSWER = (
    "If you or someone is in danger, call 911 now. Contact your barangay or local DRRMO for rescue.\n"
    "Move to higher ground if water is rising. Do not walk or drive through floodwater.\n"
    "Kung nasa panganib, tumawag sa 911 ngayon. Makipag-ugnayan sa barangay o lokal na DRRMO para sa rescue."
)

ERROR_ANSWER = (
    "Sorry, something went wrong. Please try again, or check pagasa.dost.gov.ph and your local DRRMO.\n"
    "Pasensya na, may problema. Subukan muli, o tingnan ang pagasa.dost.gov.ph at ang inyong lokal na DRRMO."
)


BAD_MEDIA_ANSWER = (
    "I could not use that file. Please send a photo or video again, or describe what you see.\n"
    "Hindi ko magamit ang file na iyon. Magpadala ulit ng larawan o video, o ilarawan ang nakikita mo.\n"
    "If someone is in danger, call 911."
)
DEFAULT_MEDIA_QUESTIONS = {"image": "Is there flood water in this photo?", "video": "Is there flood water in this video?"}

# Files for the media check may only come from this folder (the upload endpoint saves them here).
MEDIA_UPLOAD_DIR_VARIABLE = "MEDIA_UPLOAD_DIR"


def media_upload_dir() -> Path:
    """The only folder a media ``path`` may point into."""
    return Path(os.getenv(MEDIA_UPLOAD_DIR_VARIABLE) or Path(tempfile.gettempdir()) / "flood_gentic_uploads").resolve()


def check_media_url(url: str) -> str:
    """Accept only https links to a public host, since our server (and Meta's) will fetch it."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not host or host.lower() == "localhost":
        raise ValueError("Only https links to a public host are allowed.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return url  # a normal host name
    if not address.is_global:
        raise ValueError("Links to private or local addresses are not allowed.")
    return url


def media_from_dict(media: dict) -> ImageInput | VideoInput:
    """Turn the small dictionary kept in the chat state into the object the media check uses.

    The dictionary is ``{"kind": "image" or "video", "url": ... or "path": ..., "estimate_depth": bool}``.
    It holds a reference, never the file itself, so the saved conversation stays small. Raises
    ``ValueError`` for anything unusable.
    """
    kind, url, path = media.get("kind"), media.get("url"), media.get("path")
    if kind not in ("image", "video") or bool(url) == bool(path):
        raise ValueError("Media needs a kind (image or video) and exactly one of url or path.")

    if url:
        url = check_media_url(url)
    else:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(media_upload_dir()) or not resolved.is_file():
            raise ValueError("Media file not found in the upload folder.")
        path = resolved

    if kind == "video":
        return VideoInput(url=url, path=path)
    return ImageInput(url=url, path=path)


class ChatState(TypedDict, total=False):
    """What the graph remembers: the conversation and where the user is going."""

    messages: Annotated[list[AnyMessage], add_messages]
    location: dict | None
    destination: dict | None
    pending_intent: str | None
    media: dict | None  # photo or video attached to the newest message; cleared once it has been checked
    pending_parts: list | None  # topics of a multi-topic message waiting for the user's location
    safe_answer: bool  # True when the newest reply is the fixed "not confident enough" text; reset by every node


def last_question(state: ChatState) -> str:
    """Return the text of the user's newest message."""
    return state["messages"][-1].content


def route_message(state: ChatState) -> str:
    """Choose the node that handles the newest message.

    1. An emergency always wins.
    2. A message with a photo or video attached goes to the media check.
    3. A message that names a place outside Metro Manila gets the "not covered" reply
       (weather and preparedness questions do not depend on the city and are still answered).
    4. A message that is only a place ("Marikina") goes back to the agent that
       asked for one, or sets the user's location if none was waiting.
    5. A message with two or more topics (a report counts as one) goes to the multi node, which answers each one.
    6. Anything else goes by its keywords. A message with its own topic is
       never hijacked by a waiting question: "hotline Pasig" is about hotlines
       even if the evacuation agent was waiting for a city.
    """
    question = last_question(state)
    intent = classify_intent(question)
    if intent == EMERGENCY:
        return EMERGENCY_NODE

    if state.get("media"):
        return FLOOD_MEDIA_NODE

    if find_outside_place(question) and intent not in TOPICS_OUTSIDE_COVERAGE_STILL_ANSWERED:
        return OUTSIDE_NODE

    if intent == GENERAL and mentions_place(question):
        pending = state.get("pending_intent")
        return NODE_FOR_INTENT.get(pending, LOCATION_NODE)
    if split_by_intent(question):
        return MULTI_NODE
    return NODE_FOR_INTENT.get(intent, DOCUMENTS_NODE)


def build_graph(
    llm: BaseChatModel,
    retriever: HybridRetriever | None = None,
    checkpointer=None,
    get_connection: Callable | None = None,
    ors_api_key: str | None = None,
):
    """Build and compile the chat graph.

    Args:
        llm: any LangChain chat model, used by the weather and document agents.
        retriever: search over the stored documents. Without one, the document
            node replies that it has no verified information.
        checkpointer: saves conversations per thread (optional).
        get_connection: returns an open database connection, for evacuation
            centers and community reports. Without it those agents say they
            cannot read the data.
        ors_api_key: OpenRouteService key; falls back to the ORS_API_KEY environment variable.
    """

    def make_context(state: ChatState) -> ChatContext:
        return ChatContext(
            location=Location.from_dict(state.get("location")),
            destination=Location.from_dict(state.get("destination")),
            get_connection=get_connection,
            extra={"ors_api_key": ors_api_key} if ors_api_key else {},
        )

    def agent_node(agent: Callable[[str, ChatContext], AgentReply]) -> Callable[[ChatState], dict]:
        """Make a node from an agent that returns an ``AgentReply``.

        An unexpected error becomes a polite fixed reply instead of crashing the request.
        """

        def node(state: ChatState) -> dict:
            try:
                reply = agent(last_question(state), make_context(state))
            except Exception as error:
                logger.exception("Agent failed: %s", error)
                return {
                    "messages": [AIMessage(content=ERROR_ANSWER)],
                    "pending_intent": None,
                    "media": None,
                    "pending_parts": None,
                    "safe_answer": False,
                }

            return state_update(reply)

        return node

    def state_update(reply: AgentReply) -> dict:
        """The state changes for an agent's reply: the message, and anything to remember."""
        # media is cleared by every node, so an attached file is never checked a second time
        update = {
            "messages": [AIMessage(content=reply.text)],
            "pending_intent": reply.pending_intent,
            "media": None,
            "pending_parts": reply.pending_parts,
            "safe_answer": reply.safe_answer,  # set on every reply so a flag never carries over to the next turn
        }
        if reply.location:
            update["location"] = reply.location.to_dict()
        if reply.destination:
            update["destination"] = reply.destination.to_dict()
        return update

    def flood_media_node(state: ChatState) -> dict:
        """Check the attached photo or video for flood water. The media is cleared so it is only checked once."""
        try:
            media = media_from_dict(state["media"])
            reply = answer_flood_media_question(
                media, make_context(state), estimate_depth=bool(state["media"].get("estimate_depth"))
            )
        except ValueError as error:
            logger.warning("Unusable media: %s", error)
            reply = AgentReply(text=BAD_MEDIA_ANSWER)
        except Exception as error:
            logger.exception("Media check failed: %s", error)
            reply = AgentReply(text=ERROR_ANSWER)
        return {**state_update(reply), "pending_intent": None}

    def emergency_reply(state: ChatState) -> AgentReply:
        """The fixed emergency text first, then the rest of the message.

        The 911 text and local numbers never wait for anything else. After them, whatever else
        the message asked is answered by the same agents as any other message: a flood report is
        saved, the nearest centers are listed, "how long will it last?" gets its answer. A place
        outside Metro Manila gets the national numbers and a plain "not covered".
        """
        question = last_question(state)
        outside_place = find_outside_place(question)
        if outside_place:
            return AgentReply(text=f"{EMERGENCY_ANSWER}\n\n{outside_coverage_text(outside_place)}")

        place = resolve_location(question) or Location.from_dict(state.get("location"))
        text = EMERGENCY_ANSWER + ("\n\n" + format_contacts(place.city) if place else "")

        parts = split_by_intent(question, min_topics=1, in_emergency=True)
        if not parts:
            return AgentReply(text=text, location=place)

        # The 911 text is already above, so the evacuation agent must not repeat its own safety steps.
        rest = multi_reply(state, parts, extra={"safety_shown": True})
        if rest.text != ERROR_ANSWER:
            text = f"{text}\n\n{rest.text}"
        return replace(rest, text=text, location=rest.location or place)

    def emergency_node(state: ChatState) -> dict:
        """The emergency node. If anything after the fixed text fails, the fixed text alone is still sent."""
        try:
            reply = emergency_reply(state)
        except Exception as error:
            logger.exception("Emergency answer failed: %s", error)
            reply = AgentReply(text=EMERGENCY_ANSWER)
        return state_update(reply)

    def answer_from_documents(question: str, context: ChatContext) -> AgentReply:
        if retriever is None:
            return safe_reply()  # nothing to search, so do not guess
        return answer_document_question(question, llm, retriever)

    def multi_reply(
        state: ChatState, parts: list[tuple[str, str]] | None = None, extra: dict | None = None
    ) -> AgentReply:
        """Answer every topic of a multi-topic message and join the answers under headings.

        ``parts`` and ``extra`` are for the emergency node, which has already split the message
        and passes notes to the agents through ``ChatContext.extra``.

        The topics run at the same time (the weather and document agents each call the
        model, so this saves most of the waiting). A topic that needs the user's location
        while none is known is left waiting: the reply asks "which city?" and keeps the
        topic in ``pending_parts``, so the next message (just the city) completes it.
        """
        question = last_question(state)
        # A new message is split afresh; a place-only reply finishes the topics left waiting.
        parts = parts or split_by_intent(question) or [tuple(part) for part in state.get("pending_parts") or []]

        place = resolve_location(question)
        base = make_context(state)
        context = replace(
            base, location=place or Location.from_dict(state.get("location")), extra={**base.extra, **(extra or {})}
        )

        ready = [part for part in parts if context.location or part[0] not in TOPICS_NEEDING_LOCATION]
        waiting = [part for part in parts if part not in ready]

        def run_part(part: tuple[str, str]) -> AgentReply | None:
            intent, part_question = part
            try:
                return topic_agents[intent](part_question, context)
            except Exception as error:
                logger.exception("Agent for %s failed: %s", intent, error)
                return None

        with ThreadPoolExecutor(max_workers=max(len(ready), 1)) as executor:
            # copy_context() carries the LangSmith run into the worker threads.
            futures = [executor.submit(contextvars.copy_context().run, run_part, part) for part in ready]
            replies = [future.result() for future in futures]

        sections, answered = [], []
        for (intent, _), reply in zip(ready, replies):
            if reply is not None:
                sections.append(f"{TOPIC_HEADINGS[intent]}\n{reply.text}")
                answered.append(reply)
        if waiting:
            names = ", ".join(TOPIC_HEADINGS[intent].split(" / ")[0] for intent, _ in waiting)
            sections.append(f"For {names}:\n{ASK_FOR_LOCATION}")
        if not sections:
            return AgentReply(text=ERROR_ANSWER)

        return AgentReply(
            text="\n\n".join(sections),
            location=place or next((reply.location for reply in answered if reply.location), None),
            destination=next((reply.destination for reply in answered if reply.destination), None),
            pending_intent=MULTI if waiting else None,
            pending_parts=[list(part) for part in waiting] or None,
            safe_answer=any(reply.safe_answer for reply in answered),
        )

    def multi_node(state: ChatState) -> dict:
        """The node for a message with several topics. An unexpected error becomes the polite fixed reply."""
        try:
            reply = multi_reply(state)
        except Exception as error:
            logger.exception("Multi-topic answer failed: %s", error)
            reply = AgentReply(text=ERROR_ANSWER)
        return state_update(reply)

    def weather_with_city(question: str, context: ChatContext) -> AgentReply:
        """The weather agent reads the city from the question, so add the user's known city to it."""
        city = context.location.city if context.location else None
        return answer_weather_question(f"{question} ({city})" if city else question, llm)

    # Which agent answers each topic of a multi-topic message.
    topic_agents = {
        REPORT: answer_report_request,
        WEATHER: weather_with_city,
        FLOOD_RISK: answer_flood_risk_question,
        PREPAREDNESS: answer_from_documents,
        EVACUATION: answer_evacuation_question,
        ROUTE: answer_route_question,
        DURATION: answer_duration_question,
        CONTACTS: answer_contacts_question,
    }

    graph = StateGraph(ChatState)
    graph.add_node(EMERGENCY_NODE, emergency_node)
    graph.add_node(REPORT_NODE, agent_node(answer_report_request))
    graph.add_node(ROUTE_NODE, agent_node(answer_route_question))
    graph.add_node(EVACUATION_NODE, agent_node(answer_evacuation_question))
    graph.add_node(CONTACTS_NODE, agent_node(answer_contacts_question))
    graph.add_node(DURATION_NODE, agent_node(answer_duration_question))
    graph.add_node(OUTSIDE_NODE, agent_node(answer_outside_coverage))
    graph.add_node(FLOOD_RISK_NODE, agent_node(answer_flood_risk_question))
    graph.add_node(LOCATION_NODE, agent_node(answer_location_message))
    graph.add_node(WEATHER_NODE, agent_node(lambda question, context: answer_weather_question(question, llm)))
    graph.add_node(DOCUMENTS_NODE, agent_node(answer_from_documents))
    graph.add_node(FLOOD_MEDIA_NODE, flood_media_node)
    graph.add_node(MULTI_NODE, multi_node)

    graph.add_conditional_edges(START, route_message, ALL_NODES)
    for node_name in ALL_NODES:
        graph.add_edge(node_name, END)

    return graph.compile(checkpointer=checkpointer)


@dataclass
class ChatReply:
    """The assistant's reply, and whether it is the fixed safe answer (the agents were not confident enough)."""

    text: str
    safe_answer: bool = False
    blocked: bool = False  # True when the guard stopped the message before the graph ran (see utils/input_guard.py)


async def ask(
    graph, question: str, thread_id: str, location: dict | None = None, media: dict | None = None
) -> str:
    """Send one user message through the graph and return the assistant's reply text.

    Use ``ask_reply`` instead to also learn whether the reply is the flagged safe answer.
    """
    return (await ask_reply(graph, question, thread_id, location, media)).text


async def ask_reply(
    graph, question: str, thread_id: str, location: dict | None = None, media: dict | None = None
) -> ChatReply:
    """Send one user message through the graph and return the reply with its safe-answer flag.

    Messages with the same ``thread_id`` belong to the same conversation.
    ``location`` is an optional GPS fix from the app: ``{"lat": 14.65, "lng": 121.10}``.
    ``media`` is an optional photo or video for this message only (see ``media_from_dict``); an empty
    ``question`` is replaced by a default one. Set the dictionary yourself on the server (for example after
    saving an upload into the upload folder), never straight from what a user typed.
    """
    # First of all, before the graph, the database or any model: a message with a secret in it or an attempt
    # to change the assistant's rules costs nothing and is not saved in the conversation.
    blocked_category = check_message(question)
    if blocked_category:
        logger.warning("Blocked a %s message in thread %s", blocked_category, thread_id)  # never the text itself
        return ChatReply(text=blocked_reply(blocked_category), blocked=True)

    config = {
        "configurable": {"thread_id": thread_id},
        # LangSmith: a readable name, a filter tag, and the conversation id to group the runs of one chat.
        "run_name": "flood-gentic-chat",
        "tags": ["flood-gentic"],
        "metadata": {"thread_id": thread_id, "has_gps": location is not None, "has_media": media is not None},
    }
    if media and not question.strip():
        question = DEFAULT_MEDIA_QUESTIONS.get(media.get("kind"), DEFAULT_MEDIA_QUESTIONS["image"])
    graph_input: dict = {"messages": [("user", question)]}
    if media:
        graph_input["media"] = media
    if location:
        graph_input["location"] = location_from_coordinates(location["lat"], location["lng"]).to_dict()
    result = await graph.ainvoke(graph_input, config)
    return ChatReply(text=result["messages"][-1].content, safe_answer=bool(result.get("safe_answer")))
