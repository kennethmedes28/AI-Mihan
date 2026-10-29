"""Document agent: answers FLOOD questions from the stored documents (RAG).

Only flood material is searched. The search filter always carries
``hazard="flood"``; general material with no hazard in its metadata (for
example a preparedness manual or the emergency contacts) still passes it.

Flow, cheapest steps first:

    question -> cached answer?  -> done
             -> hybrid search (vector + BM25) -> rerank -> rules
             -> nothing trustworthy found?  -> fixed reply (no model call)
             -> over the hourly model budget? -> busy reply (no model call)
             -> model answers from the numbered sources, with citations

The chat model and the retriever are passed in, so this file depends on no
particular provider or database. That includes how the documents were read:
scanned PDFs (such as the NDRRMC IRR of RA 12287) are OCR'd or embedded as page
images (ColPali) at ingestion time, and this agent only sees the resulting
chunks. A chunk with no readable text is skipped here rather than shown as an
empty source.

Example:
    retriever = HybridRetriever.from_database(conn, embed_query)
    answer = answer_document_question("Ano ang dapat ihanda bago bumaha?", llm, retriever)
"""

import logging
from dataclasses import replace

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from prompts.document_prompt import DOCUMENT_PROMPT
from src.rag.models import Chunk, SearchFilter
from src.rag.reranking import rerank
from src.rag.retriever import HybridRetriever
from src.rag.rules import apply_rules
from utils.call_budget import BUSY_ANSWER, model_budget
from utils.text import normalize_question
from utils.ttl_cache import MISSING, TTLCache
from utils.tracing import traced_agent

logger = logging.getLogger(__name__)

FLOOD_HAZARD = "flood"  # the only hazard this agent answers about

RERANK_TOP_N = 8  # candidates kept after reranking
SOURCES_FOR_MODEL = 4  # chunks sent to the model; fewer means fewer tokens billed
ANSWER_CACHE_SECONDS = 30 * 60  # demo setting

# Fixed reply when nothing trustworthy was found (no model call, so no cost).
NO_EVIDENCE_ANSWER = (
    "I have no verified flood information about that. "
    "Check pagasa.dost.gov.ph, ndrrmc.gov.ph or your local DRRMO.\n"
    "Wala akong na-verify na impormasyon tungkol sa baha na ito. "
    "Tingnan ang pagasa.dost.gov.ph, ndrrmc.gov.ph o ang inyong lokal na DRRMO."
)

# Fixed reply when the chat model itself fails (not cached, so the next try can succeed).
MODEL_ERROR_ANSWER = (
    "I could not prepare an answer just now. Please try again, "
    "or call 911 if you are in danger.\n"
    "Hindi ako makabuo ng sagot ngayon. Subukan muli, "
    "o tumawag sa 911 kung nasa panganib kayo."
)

# Answers to repeated questions, kept briefly to avoid repeat model calls.
answer_cache = TTLCache(ttl_seconds=ANSWER_CACHE_SECONDS)


def flood_filter(search_filter: SearchFilter | None = None) -> SearchFilter:
    """Return the caller's filter with the hazard forced to flood.

    The city and time limits of the caller are kept. The hazard is always
    overwritten, so this agent can never be pointed at another hazard.
    """
    if search_filter is None:
        return SearchFilter(hazard=FLOOD_HAZARD)
    return replace(search_filter, hazard=FLOOD_HAZARD)


def has_readable_text(chunk: Chunk) -> bool:
    """False for chunks with no text, such as a scanned page where OCR found nothing."""
    return bool(chunk.content and chunk.content.strip())


def find_sources(
    question: str,
    retriever: HybridRetriever,
    search_filter: SearchFilter | None = None,
    min_score: float | None = None,
) -> list[Chunk]:
    """Search flood documents, rerank, and apply the safety rules.

    An empty list means "no evidence".

    ``min_score`` drops weak matches. Its right value depends on the reranker
    (0..1 with the cross-encoder), so tune it on the evaluation set.
    """
    candidates = retriever.search(question, search_filter=flood_filter(search_filter))
    candidates = [chunk for chunk in candidates if has_readable_text(chunk)]
    reranked = rerank(question, candidates, top_n=RERANK_TOP_N)
    return apply_rules(reranked, min_score=min_score, top_n=SOURCES_FOR_MODEL)


def format_sources(chunks: list[Chunk]) -> str:
    """Write the chunks as numbered sources for the prompt.

    Each label shows who issued the chunk, how far to trust it, when it was
    issued, and its page if known, so the model can tell official material
    from community reports.
    """
    lines = []
    for number, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        page = metadata.get("page")
        label_parts = [
            metadata.get("agency"),
            metadata.get("trust_tier"),
            (metadata.get("issued_at") or "")[:10],
            f"page {page}" if page else None,
        ]
        label = " | ".join(str(part) for part in label_parts if part)
        lines.append(f"[{number}] ({label})\n{chunk.content.strip()}")
    return "\n\n".join(lines)


@traced_agent("document_agent")
def answer_document_question(
    question: str,
    llm: BaseChatModel,
    retriever: HybridRetriever,
    search_filter: SearchFilter | None = None,
    min_score: float | None = None,
) -> str:
    """Answer a flood question from the stored documents, with numbered citations."""
    # The same question can have different answers per city, so the city is part of the key.
    city = search_filter.city if search_filter else None
    cache_key = (normalize_question(question), (city or "").lower())
    cached_answer = answer_cache.get(cache_key)
    if cached_answer is not MISSING:
        return cached_answer

    try:
        sources = find_sources(question, retriever, search_filter, min_score)
    except Exception as error:  # Database down, model not installed, and so on.
        logger.error("Document search failed: %s", error)
        return NO_EVIDENCE_ANSWER

    if not sources:
        return NO_EVIDENCE_ANSWER

    if not model_budget.try_use():  # hourly cap on model calls (see utils/call_budget.py)
        return BUSY_ANSWER

    chain = DOCUMENT_PROMPT | llm | StrOutputParser()
    try:
        answer = chain.invoke({"question": question, "sources": format_sources(sources)})
    except Exception as error:  # Provider outage, rate limit, and so on.
        logger.error("Document answer failed: %s", error)
        return MODEL_ERROR_ANSWER

    if answer.strip():
        answer_cache.set(cache_key, answer)
    return answer
