"""Document agent: answers questions from the stored documents (RAG).

Flow, cheapest steps first:

    question -> cached answer?  -> done
             -> hybrid search (vector + BM25) -> rerank -> rules
             -> nothing trustworthy found?  -> fixed reply (no model call)
             -> over the hourly model budget? -> busy reply (no model call)
             -> model answers from the numbered sources, with citations

The chat model and the retriever are passed in, so this file depends on no
particular provider or database.

Example:
    retriever = HybridRetriever.from_database(conn, embed_query)
    answer = answer_document_question("Ano ang dapat ihanda bago bumagyo?", llm, retriever)
"""

import logging

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

logger = logging.getLogger(__name__)

RERANK_TOP_N = 8  # candidates kept after reranking
SOURCES_FOR_MODEL = 4  # chunks sent to the model; fewer means fewer tokens billed
ANSWER_CACHE_SECONDS = 30 * 60  # demo setting

# Fixed reply when nothing trustworthy was found (no model call, so no cost).
NO_EVIDENCE_ANSWER = (
    "I have no verified information about that. "
    "Check pagasa.dost.gov.ph, ndrrmc.gov.ph or your local DRRMO.\n"
    "Wala akong na-verify na impormasyon tungkol dito. "
    "Tingnan ang pagasa.dost.gov.ph, ndrrmc.gov.ph o ang inyong lokal na DRRMO."
)

# Answers to repeated questions, kept briefly to avoid repeat model calls.
answer_cache = TTLCache(ttl_seconds=ANSWER_CACHE_SECONDS)


def find_sources(
    question: str,
    retriever: HybridRetriever,
    search_filter: SearchFilter | None = None,
    min_score: float | None = None,
) -> list[Chunk]:
    """Search, rerank, and apply the safety rules. An empty list means "no evidence".

    ``min_score`` drops weak matches. Its right value depends on the reranker
    (0..1 with the cross-encoder), so tune it on the evaluation set.
    """
    candidates = retriever.search(question, search_filter=search_filter)
    reranked = rerank(question, candidates, top_n=RERANK_TOP_N)
    return apply_rules(reranked, min_score=min_score, top_n=SOURCES_FOR_MODEL)


def format_sources(chunks: list[Chunk]) -> str:
    """Write the chunks as numbered sources for the prompt."""
    lines = []
    for number, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        label = " | ".join(
            str(value)
            for value in (metadata.get("agency"), metadata.get("trust_tier"), (metadata.get("issued_at") or "")[:10])
            if value
        )
        lines.append(f"[{number}] ({label})\n{chunk.content}")
    return "\n\n".join(lines)


def answer_document_question(
    question: str,
    llm: BaseChatModel,
    retriever: HybridRetriever,
    search_filter: SearchFilter | None = None,
    min_score: float | None = None,
) -> str:
    """Answer a question from the stored documents, with numbered citations."""
    cache_key = normalize_question(question)
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
    answer = chain.invoke({"question": question, "sources": format_sources(sources)})
    answer_cache.set(cache_key, answer)
    return answer
