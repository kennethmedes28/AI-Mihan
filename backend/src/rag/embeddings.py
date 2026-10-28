"""Turn text into embeddings (lists of numbers) for vector search.

Uses a local ``sentence-transformers`` model, so there is no per-call cost.
The model is large, so the library is optional and loaded on first use:

    uv sync --extra rag

The default model, intfloat/multilingual-e5-large, understands English and
Filipino and produces 1024 numbers per text (``EMBEDDING_DIM`` in
vector_store.py must match). It scored best in notebooks/embedding_readiness.ipynb.
Choose another model with the EMBEDDING_MODEL environment variable.

E5 models expect "query: " / "passage: " in front of each text; this module adds
them for any model whose name contains "e5". Changing the model means every
stored chunk must be embedded again.
"""

import os

from utils.ttl_cache import ttl_cache

DEFAULT_MODEL = "intfloat/multilingual-e5-large"
QUERY_CACHE_SECONDS = 30 * 60

_model = None  # loaded once, on first use


def model_name() -> str:
    """Name of the model in use."""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)


def add_prefix(texts: list[str], is_query: bool) -> list[str]:
    """Put the "query: " or "passage: " marker E5 models expect; other models are unchanged."""
    if "e5" not in model_name().lower():
        return texts
    prefix = "query: " if is_query else "passage: "
    return [prefix + text for text in texts]


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
        _model = SentenceTransformer(model_name())
    return _model


def embed_texts(texts: list[str], model=None, is_query: bool = False) -> list[list[float]]:
    """Embed many texts at once (document chunks unless ``is_query``).

    ``model`` is only passed in by tests.
    """
    if not texts:
        return []
    model = model or get_model()
    vectors = model.encode(add_prefix(texts, is_query), normalize_embeddings=True)
    return [[float(number) for number in vector] for vector in vectors]


@ttl_cache(ttl_seconds=QUERY_CACHE_SECONDS)
def embed_query(query: str) -> list[float]:
    """Embed one question. Repeated questions reuse the cached result."""
    return embed_texts([query], is_query=True)[0]
