"""Combine several ranked result lists into one (Reciprocal Rank Fusion).

Each list gives a chunk the score ``weight / (RRF_K + rank)``, and a chunk's
scores are added across lists. Only ranks are used, so BM25 scores and vector
similarities never need to be put on the same scale.
"""

from dataclasses import replace

from src.rag.models import Chunk

RRF_K = 60  # standard constant; larger values flatten the gap between ranks
EXACT_IDENTIFIER_BM25_WEIGHT = 1.5


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    weights: list[float] | None = None,
    top_n: int = 40,
) -> list[Chunk]:
    """Merge ranked lists (best first) into one list, best first.

    Args:
        ranked_lists: one list of chunks per retriever.
        weights: one weight per list; defaults to 1.0 for each.
        top_n: how many fused chunks to return.
    """
    weights = weights or [1.0] * len(ranked_lists)

    scores: dict[str, float] = {}
    first_seen: dict[str, Chunk] = {}

    for chunks, weight in zip(ranked_lists, weights):
        for rank, chunk in enumerate(chunks, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (RRF_K + rank)
            first_seen.setdefault(chunk.id, chunk)

    best_first = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [replace(first_seen[chunk_id], score=scores[chunk_id]) for chunk_id in best_first]


def bm25_weight_for(query: str) -> float:
    """Give keyword search more say when the query has an exact identifier.

    A digit (signal number, phone number, "Signal No. 3") is the cue used here.
    Names such as typhoons or barangays are not detected yet.
    """
    has_digit = any(character.isdigit() for character in query)
    return EXACT_IDENTIFIER_BM25_WEIGHT if has_digit else 1.0
