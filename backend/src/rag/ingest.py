"""Ingestion: documents -> chunks -> embeddings -> database.

    from src.rag.ingest import ingest_documents
    from src.rag.loaders import load_directory, load_pagasa_documents

    ingest_documents(conn, load_directory("data/manuals") + load_pagasa_documents())

Safe to re-run: only new or changed chunks are embedded and written.
Chunks of a document that got shorter are not deleted.
"""

from typing import Callable

from src.rag import embeddings, vector_store
from src.rag.chunking import chunk_document
from src.rag.models import Document


def ingest_documents(
    conn,
    documents: list[Document],
    embed: Callable[[list[str]], list[list[float]]] = embeddings.embed_texts,
) -> dict:
    """Store documents for search and return ``{"chunks": total, "embedded": changed}``."""
    vector_store.ensure_schema(conn)

    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    stored_hashes = vector_store.get_stored_hashes(conn, sorted({chunk.doc_id for chunk in chunks}))
    changed = vector_store.chunks_needing_embedding(chunks, stored_hashes)

    if changed:
        texts = [chunk.content for chunk in changed]
        vector_store.upsert_chunks(
            conn,
            changed,
            embed(texts),
            [vector_store.content_hash(text) for text in texts],
        )

    return {"chunks": len(chunks), "embedded": len(changed)}
