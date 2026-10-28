"""Read and write chunks in Supabase Postgres (pgvector).

Every function takes an open psycopg connection, so the caller controls the
connection and tests can pass a fake. Connections are expected to use
``dict_row`` rows (see ``checkpointer/db_config.py``).

Embeddings are sent as text such as ``[0.1,0.2]`` and cast with ``::vector``,
which avoids needing the ``pgvector`` Python package.
"""

import hashlib
import json

from src.rag.models import Chunk, SearchFilter

# Must match the embedding model (bge-m3 and multilingual-e5-large produce 1024 numbers per text).
EMBEDDING_DIM = 1024

CREATE_SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    doc_id       text    NOT NULL,
    chunk_idx    integer NOT NULL,
    content      text    NOT NULL,
    content_hash text    NOT NULL,
    embedding    vector({EMBEDDING_DIM}) NOT NULL,
    metadata     jsonb   NOT NULL DEFAULT '{{}}',
    PRIMARY KEY (doc_id, chunk_idx)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

UPSERT_SQL = """
INSERT INTO chunks (doc_id, chunk_idx, content, content_hash, embedding, metadata)
VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)
ON CONFLICT (doc_id, chunk_idx) DO UPDATE SET
    content      = EXCLUDED.content,
    content_hash = EXCLUDED.content_hash,
    embedding    = EXCLUDED.embedding,
    metadata     = EXCLUDED.metadata
WHERE chunks.content_hash <> EXCLUDED.content_hash
"""


def _vector_text(embedding: list[float]) -> str:
    """Write an embedding the way pgvector reads it: ``[0.1,0.2,...]``."""
    return "[" + ",".join(str(number) for number in embedding) + "]"


def _filter_sql(search_filter: SearchFilter | None) -> tuple[str, list]:
    """Build the WHERE clause and its values for a metadata filter.

    Chunks with no hazard / city / valid_until pass, matching ``SearchFilter.matches``.
    """
    if search_filter is None:
        return "TRUE", []

    conditions = [
        "(metadata->>'valid_until' IS NULL OR (metadata->>'valid_until')::timestamptz >= %s)"
    ]
    values: list = [search_filter.current_time()]

    for name, wanted in (("hazard", search_filter.hazard), ("city", search_filter.city)):
        if wanted:
            conditions.append(f"(metadata->>'{name}' IS NULL OR lower(metadata->>'{name}') = lower(%s))")
            values.append(wanted)

    return " AND ".join(conditions), values


def _row_to_chunk(row: dict, score: float = 0.0) -> Chunk:
    return Chunk(
        doc_id=row["doc_id"],
        chunk_idx=row["chunk_idx"],
        content=row["content"],
        metadata=row["metadata"] or {},
        score=score,
    )


def content_hash(text: str) -> str:
    """Fingerprint of a chunk's text, used to tell whether it changed."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_stored_hashes(conn, doc_ids: list[str]) -> dict[tuple[str, int], str]:
    """Return ``{(doc_id, chunk_idx): content_hash}`` for the given documents."""
    rows = conn.execute(
        "SELECT doc_id, chunk_idx, content_hash FROM chunks WHERE doc_id = ANY(%s)",
        [doc_ids],
    ).fetchall()
    return {(row["doc_id"], row["chunk_idx"]): row["content_hash"] for row in rows}


def chunks_needing_embedding(
    chunks: list[Chunk], stored_hashes: dict[tuple[str, int], str]
) -> list[Chunk]:
    """Keep only new or changed chunks.

    Embedding is the paid / slow step of ingestion. Call this before embedding
    so a re-run of ingestion only embeds text that is actually different.
    """
    return [
        chunk
        for chunk in chunks
        if stored_hashes.get((chunk.doc_id, chunk.chunk_idx)) != content_hash(chunk.content)
    ]


def ensure_schema(conn) -> None:
    """Create the extension, table, and index if they do not exist yet."""
    conn.execute(CREATE_SCHEMA_SQL)


def upsert_chunks(conn, chunks: list[Chunk], embeddings: list[list[float]], hashes: list[str]) -> None:
    """Insert chunks, or update them when their content changed.

    Re-running ingestion is safe: a chunk whose ``content_hash`` is unchanged is left alone.
    """
    for chunk, embedding, content_hash in zip(chunks, embeddings, hashes):
        conn.execute(
            UPSERT_SQL,
            (
                chunk.doc_id,
                chunk.chunk_idx,
                chunk.content,
                content_hash,
                _vector_text(embedding),
                json.dumps(chunk.metadata),
            ),
        )


def search_dense(
    conn, query_embedding: list[float], k: int = 30, search_filter: SearchFilter | None = None
) -> list[Chunk]:
    """Return the ``k`` chunks closest to the query embedding, best first."""
    where, values = _filter_sql(search_filter)
    vector = _vector_text(query_embedding)

    sql = f"""
        SELECT doc_id, chunk_idx, content, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        WHERE {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    rows = conn.execute(sql, [vector, *values, vector, k]).fetchall()
    return [_row_to_chunk(row, score=float(row["similarity"])) for row in rows]


def load_all_chunks(conn) -> list[Chunk]:
    """Load every chunk (without embeddings), for building the BM25 index."""
    rows = conn.execute("SELECT doc_id, chunk_idx, content, metadata FROM chunks").fetchall()
    return [_row_to_chunk(row) for row in rows]
