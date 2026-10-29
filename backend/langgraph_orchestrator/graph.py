"""The chat graph: routes each message to the right flood agent.

    START --> (route by keywords) --+--> emergency    fixed reply + local hotlines, no model call
                                    +--> report       save a community flood report
                                    +--> route        route to an evacuation center that avoids floods
                                    +--> evacuation   nearest evacuation centers that are not flooded
                                    +--> contacts     hotlines for the user's city
                                    +--> location    a place name alone ("Marikina") sets where the user is
                                    +--> flood_risk   rule-based flood risk for a city
                                    +--> weather      PAGASA + Open-Meteo
                                    +--> documents    stored flood material (RAG)
                                    |
                                    +--> END

Every node adds one assistant message to the conversation. The graph also
remembers, per conversation:

    * ``location``       where the user is
    * ``destination``    the evacuation center last suggested, for "how do I get there?"
    * ``pending_intent`` set when an agent asked "which city are you in?", so the
                         next message (just "Marikina") goes back to that agent

When a checkpointer is given, the conversation is saved per ``thread_id``
(see checkpointer/postgres_checkpointer.py).

Example:
    graph = build_graph(llm, retriever, checkpointer, get_connection=get_connection)
    reply = await ask(graph, "Saan ang pinakamalapit na evacuation center?", "user-123",
                      location={"lat": 14.65, "lng": 121.10})
"""

import logging
from typing import Annotated, Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.contacts_agent import answer_contacts_question, format_contacts
from agents.document_agents import NO_EVIDENCE_ANSWER, answer_document_question
from agents.evacuation_agent import answer_evacuation_question
from agents.flood_risk_agent import answer_flood_risk_question
from agents.flood_types import AgentReply, ChatContext, Location
from agents.location_agent import answer_location_message, location_from_coordinates, mentions_place, resolve_location
from agents.report_agent import answer_report_request
from agents.route_agent import answer_route_question
from agents.weather_agent import answer_weather_question
from langgraph_orchestrator.intents import (
    CONTACTS,
    EMERGENCY,
    EVACUATION,
    FLOOD_RISK,
    GENERAL,
    PREPAREDNESS,
    REPORT,
    ROUTE,
    WEATHER,
    classify_intent,
)
from src.rag.retriever import HybridRetriever

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
ALL_NODES = [
    EMERGENCY_NODE, REPORT_NODE, ROUTE_NODE, EVACUATION_NODE,
    CONTACTS_NODE, FLOOD_RISK_NODE, WEATHER_NODE, DOCUMENTS_NODE, LOCATION_NODE,
]

# Which node answers each intent. Preparedness and anything unrecognised go to the documents.
NODE_FOR_INTENT = {
    EMERGENCY: EMERGENCY_NODE,
    REPORT: REPORT_NODE,
    ROUTE: ROUTE_NODE,
    EVACUATION: EVACUATION_NODE,
    CONTACTS: CONTACTS_NODE,
    FLOOD_RISK: FLOOD_RISK_NODE,
    WEATHER: WEATHER_NODE,
    PREPAREDNESS: DOCUMENTS_NODE,
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


class ChatState(TypedDict, total=False):
    """What the graph remembers: the conversation and where the user is going."""

    messages: Annotated[list[AnyMessage], add_messages]
    location: dict | None
    destination: dict | None
    pending_intent: str | None


def last_question(state: ChatState) -> str:
    """Return the text of the user's newest message."""
    return state["messages"][-1].content


def route_message(state: ChatState) -> str:
    """Choose the node that handles the newest message.

    1. An emergency always wins.
    2. A message that is only a place ("Marikina") goes back to the agent that
       asked for one, or sets the user's location if none was waiting.
    3. Anything else goes by its keywords. A message with its own topic is
       never hijacked by a waiting question: "hotline Pasig" is about hotlines
       even if the evacuation agent was waiting for a city.
    """
    question = last_question(state)
    intent = classify_intent(question)
    if intent == EMERGENCY:
        return EMERGENCY_NODE

    if intent == GENERAL and mentions_place(question):
        pending = state.get("pending_intent")
        return NODE_FOR_INTENT.get(pending, LOCATION_NODE)
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
                return {"messages": [AIMessage(content=ERROR_ANSWER)], "pending_intent": None}

            update = {"messages": [AIMessage(content=reply.text)], "pending_intent": reply.pending_intent}
            if reply.location:
                update["location"] = reply.location.to_dict()
            if reply.destination:
                update["destination"] = reply.destination.to_dict()
            return update

        return node

    def text_node(answer_function: Callable[[str], str]) -> Callable[[ChatState], dict]:
        """Make a node from a function that only returns text (the weather and document agents)."""
        return agent_node(lambda question, context: AgentReply(text=answer_function(question)))

    def emergency(question: str, context: ChatContext) -> AgentReply:
        """The fixed emergency text, plus local numbers when the city is known."""
        place = resolve_location(question) or context.location
        text = EMERGENCY_ANSWER + ("\n\n" + format_contacts(place.city) if place else "")
        return AgentReply(text=text, location=place)

    def answer_from_documents(question: str) -> str:
        if retriever is None:
            return NO_EVIDENCE_ANSWER
        return answer_document_question(question, llm, retriever)

    graph = StateGraph(ChatState)
    graph.add_node(EMERGENCY_NODE, agent_node(emergency))
    graph.add_node(REPORT_NODE, agent_node(answer_report_request))
    graph.add_node(ROUTE_NODE, agent_node(answer_route_question))
    graph.add_node(EVACUATION_NODE, agent_node(answer_evacuation_question))
    graph.add_node(CONTACTS_NODE, agent_node(answer_contacts_question))
    graph.add_node(FLOOD_RISK_NODE, agent_node(answer_flood_risk_question))
    graph.add_node(LOCATION_NODE, agent_node(answer_location_message))
    graph.add_node(WEATHER_NODE, text_node(lambda question: answer_weather_question(question, llm)))
    graph.add_node(DOCUMENTS_NODE, text_node(answer_from_documents))

    graph.add_conditional_edges(START, route_message, ALL_NODES)
    for node_name in ALL_NODES:
        graph.add_edge(node_name, END)

    return graph.compile(checkpointer=checkpointer)


async def ask(graph, question: str, thread_id: str, location: dict | None = None) -> str:
    """Send one user message through the graph and return the assistant's reply.

    Messages with the same ``thread_id`` belong to the same conversation.
    ``location`` is an optional GPS fix from the app: ``{"lat": 14.65, "lng": 121.10}``.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        # LangSmith: a readable name, a filter tag, and the conversation id to group the runs of one chat.
        "run_name": "flood-gentic-chat",
        "tags": ["flood-gentic"],
        "metadata": {"thread_id": thread_id, "has_gps": location is not None},
    }
    graph_input: dict = {"messages": [("user", question)]}
    if location:
        graph_input["location"] = location_from_coordinates(location["lat"], location["lng"]).to_dict()
    result = await graph.ainvoke(graph_input, config)
    return result["messages"][-1].content
