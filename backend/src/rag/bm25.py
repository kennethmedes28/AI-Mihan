"""Keyword search (BM25) over all chunks, kept in memory.

BM25 finds exact words (typhoon names, "Signal No. 3", place names) that
embedding search can miss. The index is built from the chunks already stored
in the database and rebuilt after each ingestion run.
"""

import re
from dataclasses import replace

import bm25s
import numpy as np

from src.rag.models import Chunk, SearchFilter


def tokenize(text: str) -> list[str]:
    """Split text into lowercase words and numbers.

    No stemming: it can mangle Filipino words. Phone numbers split on dashes,
    so "8888-1234" becomes "8888" and "1234".
    """
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """BM25 index over a fixed list of chunks."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = None

        if chunks:
            self._bm25 = bm25s.BM25()
            self._bm25.index([tokenize(chunk.content) for chunk in chunks], show_progress=False)

    def search(self, query: str, k: int = 30, search_filter: SearchFilter | None = None) -> list[Chunk]:
        """Return up to ``k`` best-matching chunks, best first.

        Chunks rejected by ``search_filter`` are excluded before ranking.
        Chunks that share no word with the query are never returned.
        """
        query_tokens = tokenize(query)
        if self._bm25 is None or not query_tokens:
            return []

        # Score every chunk, then drop filtered-out ones by giving them no score.
        scores = np.array(self._bm25.get_scores(query_tokens), dtype=float)
        if search_filter is not None:
            allowed = np.array([search_filter.matches(chunk.metadata) for chunk in self.chunks])
            scores = np.where(allowed, scores, 0.0)

        best_first = np.argsort(scores)[::-1][:k]
        return [
            replace(self.chunks[index], score=float(scores[index]))
            for index in best_first
            if scores[index] > 0
        ]
