"""Turn text into embeddings (lists of numbers) for vector search.

Uses a local ``sentence-transformers`` model, so there is no per-call cost.
The model is large, so the library is optional and loaded on first use:

    uv sync --extra rag

The default model, BAAI/bge-m3, understands English and Filipino and produces
1024 numbers per text (``EMBEDDING_DIM`` in vector_store.py must match).
Choose another model with the EMBEDDING_MODEL environment variable.
"""

import os

from utils.ttl_cache import ttl_cache

DEFAULT_MODEL = "BAAI/bge-m3"
QUERY_CACHE_SECONDS = 30 * 60

_model = None  # loaded once, on first use


def get_model():
    """Load the embedding model the first time it is needed."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: uv sync --extra rag"
            ) from error
        _model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    return _model


def embed_texts(texts: list[str], model=None) -> list[list[float]]:
    """Embed many texts at once. ``model`` is only passed in by tests."""
    if not texts:
        return []
    model = model or get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [[float(number) for number in vector] for vector in vectors]


@ttl_cache(ttl_seconds=QUERY_CACHE_SECONDS)
def embed_query(query: str) -> list[float]:
    """Embed one question. Repeated questions reuse the cached result."""
    return embed_texts([query])[0]
