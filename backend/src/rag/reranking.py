"""Re-order candidate chunks by reading each one together with the question.

Vector search and BM25 are fast but rough. A cross-encoder reads the question
and a chunk side by side and scores how well the chunk answers it, which is
more accurate (see section 4 of docs/ingestion_details.md).

The model is large, so the library is optional and loaded on first use:

    uv sync --extra rag

Set RERANKER_MODEL=none to skip reranking entirely (cheapest option for demos):
candidates then keep their fusion order and fusion scores.
"""

import math
import os
from dataclasses import replace
from typing import Callable

from src.rag.models import Chunk

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

_model = None  # loaded once, on first use


def get_cross_encoder():
    """Load the reranking model the first time it is needed."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: uv sync --extra rag"
            ) from error
        _model = CrossEncoder(os.getenv("RERANKER_MODEL", DEFAULT_MODEL))
    return _model


def _sigmoid(value: float) -> float:
    """Squash a raw model score into 0..1, so a fixed threshold makes sense."""
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def score_pairs(query: str, texts: list[str]) -> list[float]:
    """Score how well each text answers the query (0..1, higher is better)."""
    raw_scores = get_cross_encoder().predict([(query, text) for text in texts])
    return [_sigmoid(score) for score in raw_scores]


def rerank(
    query: str,
    chunks: list[Chunk],
    top_n: int = 8,
    scorer: Callable[[str, list[str]], list[float]] | None = None,
) -> list[Chunk]:
    """Return the ``top_n`` chunks that best answer the query, best first.

    ``scorer`` is only passed in by tests; by default the cross-encoder is used.
    """
    if not chunks:
        return []

    if scorer is None and os.getenv("RERANKER_MODEL", "").lower() == "none":
        return chunks[:top_n]

    scores = (scorer or score_pairs)(query, [chunk.content for chunk in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [replace(chunk, score=float(score)) for chunk, score in ranked[:top_n]]
