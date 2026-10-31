"""Document agent: answers FLOOD questions from the stored documents (RAG).

Only flood material is searched. The search filter always carries
``hazard="flood"``; general material with no hazard in its metadata (for
example a preparedness manual or the emergency contacts) still passes it.

Flow, cheapest steps first:

    question -> cached answer?  -> done
             -> hybrid search (vector + BM25) -> rerank -> rules
             -> only weak matches, or none? -> search trusted websites in a browser (web_lookup_agent.py)
             -> websites have nothing, or the matches are almost unrelated? -> the safe answer
                (fixed safety text, flagged; no model call, so nothing can be invented)
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
import re
from dataclasses import replace

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from agents.flood_types import AgentReply
from agents.safe_answer import safe_reply
from agents.web_lookup_agent import answer_from_web
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

# Best rerank score (after the trust boost). A starting point: tune both values on the evaluation set.
#   at or above LOW_CONFIDENCE_SCORE   the documents answer
#   from VERY_LOW to LOW (or no match) the trusted websites are searched instead
#   below VERY_LOW_CONFIDENCE_SCORE    the matches are noise: no model and no web text, only the safe answer
LOW_CONFIDENCE_SCORE = 0.3
VERY_LOW_CONFIDENCE_SCORE = 0.1

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


def is_low_confidence(sources: list[Chunk]) -> bool:
    """True when there are no sources, or even the best one is a weak match."""
    return not sources or max(chunk.score for chunk in sources) < LOW_CONFIDENCE_SCORE


def is_very_low_confidence(sources: list[Chunk]) -> bool:
    """True when sources were found but even the best one is almost unrelated to the question.

    No sources at all is not "very low": nothing was found, so the websites are still worth a try.
    """
    return bool(sources) and max(chunk.score for chunk in sources) < VERY_LOW_CONFIDENCE_SCORE


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


def format_source_list(chunks: list[Chunk], answer: str) -> str:
    """List the documents behind the answer, so "[3]" in the text can be looked up.

    Only the sources the answer cites are listed. When it cites none (for example "I have no
    verified information"), nothing is listed: the model did not rely on them.
    """
    cited = sorted({int(number) for number in re.findall(r"\[(\d+)\]", answer)})
    lines = []
    for number in cited:
        if not 1 <= number <= len(chunks):
            continue
        metadata = chunks[number - 1].metadata
        page = metadata.get("page")
        label_parts = [
            metadata.get("title") or metadata.get("source"),
            metadata.get("agency"),
            (metadata.get("issued_at") or "")[:10],
            f"page {page}" if page else None,
        ]
        lines.append(f"[{number}] " + ", ".join(str(part) for part in label_parts if part))
    return "Sources / Mga pinagkunan:\n" + "\n".join(lines) if lines else ""


@traced_agent("document_agent")
def answer_document_question(
    question: str,
    llm: BaseChatModel,
    retriever: HybridRetriever,
    search_filter: SearchFilter | None = None,
    min_score: float | None = None,
) -> AgentReply:
    """Answer a flood question from the stored documents, with numbered citations.

    When the evidence is too weak the reply is the fixed safe answer (``safe_answer=True``),
    never a model guess.
    """
    # The same question can have different answers per city, so the city is part of the key.
    city = search_filter.city if search_filter else None
    cache_key = (normalize_question(question), (city or "").lower())
    cached_answer = answer_cache.get(cache_key)
    if cached_answer is not MISSING:
        return AgentReply(text=cached_answer)

    try:
        sources = find_sources(question, retriever, search_filter, min_score)
    except Exception as error:  # Database down, model not installed, and so on.
        logger.error("Document search failed: %s", error)
        sources = []

    if is_very_low_confidence(sources):
        logger.info("Very low document confidence; giving the safe answer.")
        return safe_reply()

    if is_low_confidence(sources):
        # The stored documents do not clearly answer this: ask the trusted websites instead.
        logger.info("Low document confidence; searching trusted websites.")
        web_answer = answer_from_web(question, llm)
        if web_answer:
            answer_cache.set(cache_key, web_answer)
            return AgentReply(text=web_answer)
        return safe_reply()

    if not model_budget.try_use():  # hourly cap on model calls (see utils/call_budget.py)
        return AgentReply(text=BUSY_ANSWER)

    chain = DOCUMENT_PROMPT | llm | StrOutputParser()
    try:
        answer = chain.invoke({"question": question, "sources": format_sources(sources)})
    except Exception as error:  # Provider outage, rate limit, and so on.
        logger.error("Document answer failed: %s", error)
        return AgentReply(text=MODEL_ERROR_ANSWER)

    if answer.strip():
        source_list = format_source_list(sources, answer)
        if source_list:
            answer = f"{answer.rstrip()}\n\n{source_list}"
        answer_cache.set(cache_key, answer)
    return AgentReply(text=answer)
