"""POST /chat: send one message through the chat graph and get the reply."""

import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from langgraph_orchestrator.graph import ask_reply
from src.api.deps import DATABASE_UNAVAILABLE, get_graph, limit_requests
from src.api.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(limit_requests("chat"))])
async def chat(body: ChatRequest, graph=Depends(get_graph)) -> ChatResponse:
    """Answer one message.

    The first call needs no ``thread_id``; the reply carries one. Send it back with every
    later message so the assistant remembers the place and the last suggested center.
    ``location`` is an optional GPS fix from the app.

    The reply is plain text. It is not yet streamed, and it carries no separate source list.
    """
    thread_id = str(body.thread_id) if body.thread_id else str(uuid4())
    location = body.location.model_dump() if body.location else None

    try:
        reply = await ask_reply(graph, body.message, thread_id=thread_id, location=location)
    except Exception:
        # The message text is left out of the log on purpose: it can hold a place and a phone number.
        logger.exception("Chat failed for thread %s", thread_id)
        raise HTTPException(status_code=502, detail=DATABASE_UNAVAILABLE)

    return ChatResponse(thread_id=thread_id, reply=reply.text, safe_answer=reply.safe_answer)
