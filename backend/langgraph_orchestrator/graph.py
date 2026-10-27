"""The chat graph: routes each message to the right agent.

    START --> (route by keywords) --+--> emergency   fixed reply, no model call
                                    +--> weather     PAGASA + Open-Meteo
                                    +--> earthquake  PHIVOLCS
                                    +--> documents   stored preparedness material (RAG)
                                    |
                                    +--> END

Every node adds one assistant message to the conversation. When a
checkpointer is given, the conversation is saved per ``thread_id``, so the
same user can continue later (see checkpointer/postgres_checkpointer.py).

Example:
    graph = build_graph(llm, retriever, checkpointer)
    reply = await ask(graph, "Uulan ba bukas sa Makati?", thread_id="user-123")
"""

import logging
from typing import Annotated, Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.document_agents import NO_EVIDENCE_ANSWER, answer_document_question
from agents.earthquake_agent import answer_earthquake_question
from agents.weather_agent import answer_weather_question
from langgraph_orchestrator.intents import EARTHQUAKE, EMERGENCY, PREPAREDNESS, WEATHER, classify_intent
from src.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Node names.
EMERGENCY_NODE = "emergency"
WEATHER_NODE = "weather"
EARTHQUAKE_NODE = "earthquake"
DOCUMENTS_NODE = "documents"

# Fixed reply for emergencies: instant, free, and does not depend on any API.
# 911 is the Philippines' national emergency number.
EMERGENCY_ANSWER = (
    "If you or someone is in danger, call 911 now. Contact your barangay or local DRRMO for rescue.\n"
    "If the ground is shaking: Drop, Cover, and Hold On. Move to higher ground if water is rising.\n"
    "Kung nasa panganib, tumawag sa 911 ngayon. Makipag-ugnayan sa barangay o lokal na DRRMO para sa rescue."
)

ERROR_ANSWER = (
    "Sorry, something went wrong. Please try again, or check pagasa.dost.gov.ph and your local DRRMO.\n"
    "Pasensya na, may problema. Subukan muli, o tingnan ang pagasa.dost.gov.ph at ang inyong lokal na DRRMO."
)


class ChatState(TypedDict):
    """What the graph remembers: the conversation so far."""

    messages: Annotated[list[AnyMessage], add_messages]


def last_question(state: ChatState) -> str:
    """Return the text of the user's newest message."""
    return state["messages"][-1].content


def route_message(state: ChatState) -> str:
    """Choose the node that handles the newest message."""
    intent = classify_intent(last_question(state))
    if intent == EMERGENCY:
        return EMERGENCY_NODE
    if intent == WEATHER:
        return WEATHER_NODE
    if intent == EARTHQUAKE:
        return EARTHQUAKE_NODE
    return DOCUMENTS_NODE  # preparedness and everything unrecognised


def build_graph(llm: BaseChatModel, retriever: HybridRetriever | None = None, checkpointer=None):
    """Build and compile the chat graph.

    Args:
        llm: any LangChain chat model, used by the weather, earthquake and document agents.
        retriever: search over the stored documents. Without one, the document
            node replies that it has no verified information.
        checkpointer: saves conversations per thread (optional).
    """

    def reply(answer_function: Callable[[str], str]) -> Callable[[ChatState], dict]:
        """Make a node that answers the newest question with ``answer_function``.

        An unexpected error becomes a polite fixed reply instead of crashing the request.
        """

        def node(state: ChatState) -> dict:
            try:
                answer = answer_function(last_question(state))
            except Exception as error:
                logger.exception("Agent failed: %s", error)
                answer = ERROR_ANSWER
            return {"messages": [AIMessage(content=answer)]}

        return node

    def answer_from_documents(question: str) -> str:
        if retriever is None:
            return NO_EVIDENCE_ANSWER
        return answer_document_question(question, llm, retriever)

    graph = StateGraph(ChatState)
    graph.add_node(EMERGENCY_NODE, reply(lambda question: EMERGENCY_ANSWER))
    graph.add_node(WEATHER_NODE, reply(lambda question: answer_weather_question(question, llm)))
    graph.add_node(EARTHQUAKE_NODE, reply(lambda question: answer_earthquake_question(question, llm)))
    graph.add_node(DOCUMENTS_NODE, reply(answer_from_documents))

    graph.add_conditional_edges(
        START,
        route_message,
        [EMERGENCY_NODE, WEATHER_NODE, EARTHQUAKE_NODE, DOCUMENTS_NODE],
    )
    for node_name in (EMERGENCY_NODE, WEATHER_NODE, EARTHQUAKE_NODE, DOCUMENTS_NODE):
        graph.add_edge(node_name, END)

    return graph.compile(checkpointer=checkpointer)


async def ask(graph, question: str, thread_id: str) -> str:
    """Send one user message through the graph and return the assistant's reply.

    Messages with the same ``thread_id`` belong to the same conversation.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke({"messages": [("user", question)]}, config)
    return result["messages"][-1].content
