"""Business rules applied after reranking (see section 4 of the design doc).

In order: drop expired chunks, boost official sources over community reports,
drop weak matches, limit chunks per document, keep the best few.

An empty result means "nothing trustworthy found": the caller should answer
with the no-verified-information fallback instead of guessing.
"""

from dataclasses import replace
from datetime import datetime

from src.rag.models import PHILIPPINE_TZ, Chunk, parse_time

# Multiplies the rerank score. Unknown tiers are left unchanged.
TRUST_BOOST = {"official": 1.2, "community": 0.8}

DEFAULT_MAX_PER_DOC = 3
DEFAULT_TOP_N = 8


def apply_rules(
    chunks: list[Chunk],
    now: datetime | None = None,
    min_score: float | None = None,
    max_per_doc: int = DEFAULT_MAX_PER_DOC,
    top_n: int = DEFAULT_TOP_N,
) -> list[Chunk]:
    """Filter and reorder reranked chunks.

    Args:
        chunks: reranked chunks; ``score`` is the reranker's score.
        now: current time, for expiry checks (defaults to the real time).
        min_score: drop chunks scoring below this after the trust boost.
            The right value depends on the reranker, so tune it on the
            evaluation set; ``None`` keeps everything.
        max_per_doc: most chunks allowed from one document.
        top_n: most chunks returned.
    """
    now = now or datetime.now(PHILIPPINE_TZ)

    # 1. Expired chunks are never shown.
    current = []
    for chunk in chunks:
        valid_until = parse_time(chunk.metadata.get("valid_until"))
        if valid_until is None or valid_until >= now:
            current.append(chunk)

    # 2. Trust boost, then best first.
    boosted = [
        replace(chunk, score=chunk.score * TRUST_BOOST.get(chunk.metadata.get("trust_tier"), 1.0))
        for chunk in current
    ]
    boosted.sort(key=lambda chunk: chunk.score, reverse=True)

    # 3. Weak matches.
    if min_score is not None:
        boosted = [chunk for chunk in boosted if chunk.score >= min_score]

    # 4. Diversity: stop one document from filling the context.
    kept = []
    per_doc: dict[str, int] = {}
    for chunk in boosted:
        if per_doc.get(chunk.doc_id, 0) >= max_per_doc:
            continue
        per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
        kept.append(chunk)

    return kept[:top_n]
