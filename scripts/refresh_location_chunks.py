"""Replace only the NCR location and evacuation-center chunks after their corpus CSV files changed.

Run from backend/ (loads the embedding model; the first run may download it):

    uv run python ../scripts/refresh_location_chunks.py --dry-run   # show the new chunks, change nothing
    uv run python ../scripts/refresh_location_chunks.py             # Supabase + a new FAISS run

Only the ``ncr-locations-<city>`` and ``evacuation-centers-<city>`` documents are embedded again (about 120 short lines).
All other chunks keep their stored vectors, so this is much faster than ingest_corpus.py.

    1. Supabase: delete the old ``ncr-locations-*`` and ``evacuation-centers-*`` rows, insert the new ones.
    2. FAISS: write a NEW folder in faiss_vectors/ made of the old run's other chunks
       (vectors copied, not re-embedded) plus the new chunks. The old folder is kept.

The PDF renamed from world-bank to MMFMP is renamed in the new FAISS run too.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_DIR / "corpus"
FAISS_DIR = REPO_DIR / "faiss_vectors"

sys.path.insert(0, str(REPO_DIR / "backend"))  # so `src` can be imported
sys.path.insert(0, str(REPO_DIR / "scripts"))  # so the rename tables can be shared
load_dotenv()

from load_to_supabase import DOC_ID_RENAMES, SOURCE_RENAMES, latest_faiss_run  # noqa: E402

# Chunks of these documents are replaced; every other chunk is kept as it is.
REFRESHED_DOC_PREFIXES = ("ncr-locations-", "evacuation-centers-")


def build_location_chunks():
    """Chunk the current corpus/ncr_locations.csv and corpus/ncr_evacuation_centers.csv."""
    from src.rag.chunking import chunk_document
    from src.rag.loaders import load_evacuation_centers_csv, load_ncr_locations_csv

    documents = load_ncr_locations_csv(CORPUS_DIR / "ncr_locations.csv")
    documents += load_evacuation_centers_csv(CORPUS_DIR / "ncr_evacuation_centers.csv")
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks


def replace_in_supabase(chunks, vectors) -> None:
    """Delete the old location and evacuation-center rows and insert the new ones in one transaction."""
    from src.rag import vector_store
    from src.services.fetch_postgre_database import get_connection

    hashes = [vector_store.content_hash(chunk.content) for chunk in chunks]
    with get_connection() as conn, conn.transaction():
        patterns = [prefix + "%" for prefix in REFRESHED_DOC_PREFIXES]
        deleted = conn.execute("DELETE FROM chunks WHERE doc_id LIKE ANY(%s)", (patterns,)).rowcount
        vector_store.upsert_chunks(conn, chunks, vectors, hashes)
    print(f"Supabase: removed {deleted} old chunks, added {len(chunks)}")


def write_new_faiss_run(new_chunks, new_vectors) -> Path:
    """Save a new run: the old run's other chunks (vectors copied) plus the new chunks."""
    from src.rag import embeddings, faiss_store
    from src.rag.models import Chunk

    index, records, _ = faiss_store.load_run(latest_faiss_run())
    old_vectors = index.reconstruct_n(0, index.ntotal).tolist()

    kept_chunks, kept_vectors = [], []
    for record, vector in zip(records, old_vectors):
        if record["doc_id"].startswith(REFRESHED_DOC_PREFIXES):
            continue  # replaced by the new chunks
        metadata = dict(record["metadata"])
        if "source" in metadata:
            metadata["source"] = SOURCE_RENAMES.get(metadata["source"], metadata["source"])
        doc_id = DOC_ID_RENAMES.get(record["doc_id"], record["doc_id"])
        kept_chunks.append(Chunk(doc_id, record["chunk_idx"], record["content"], metadata))
        kept_vectors.append(vector)

    source_files = sorted(path for path in CORPUS_DIR.iterdir() if path.suffix.lower() in {".pdf", ".csv"})
    return faiss_store.save_run(
        FAISS_DIR, kept_chunks + new_chunks, kept_vectors + list(new_vectors), embeddings.model_name(), source_files
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="show the new chunks, change nothing")
    args = parser.parse_args()

    chunks = build_location_chunks()
    for chunk in chunks:
        print(f"{chunk.id}: {chunk.content}")
    print(f"{len(chunks)} chunks")
    if args.dry_run:
        return

    from src.rag import embeddings

    vectors = embeddings.embed_texts([chunk.content for chunk in chunks])
    replace_in_supabase(chunks, vectors)
    print(f"Saved FAISS run: {write_new_faiss_run(chunks, vectors)}")


if __name__ == "__main__":
    main()
