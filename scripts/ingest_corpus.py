"""Chunk, embed and store the PDFs and the two CSV files in corpus/ (see docs/ingestion_details.md).

Run from the backend folder so imports and .env are found:

    cd backend
    uv run python ../scripts/ingest_corpus.py --dry-run     # chunk only: no model, no output
    uv run python ../scripts/ingest_corpus.py               # embed, save a new folder in faiss_vectors/
    uv run python ../scripts/ingest_corpus.py --supabase    # also write the chunks to Supabase

Each FAISS run goes into its own timestamped folder, so earlier runs are kept.
The Supabase step needs SUPABASE_DB_URL in backend/.env.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_DIR / "corpus"
FAISS_DIR = REPO_DIR / "faiss_vectors"

load_dotenv()  # picks up backend/.env, HF_TOKEN and SUPABASE_DB_URL


def load_corpus_documents():
    """Load every PDF and the two CSV files in corpus/. Scanned pages are read with OCR (slow)."""
    from src.rag.loaders import load_directory, load_emergency_contacts_csv, load_ncr_locations_csv

    documents = load_directory(CORPUS_DIR)
    documents += load_emergency_contacts_csv(CORPUS_DIR / "ph_emergency_contacts.csv")
    documents += load_ncr_locations_csv(CORPUS_DIR / "ncr_locations.csv")
    # Skip a second copy of the same text (for example a renamed duplicate PDF).
    unique, seen_texts = [], set()
    for document in documents:
        if document.text in seen_texts:
            print(f"Skipping duplicate: {document.metadata['source']}")
            continue
        seen_texts.add(document.text)
        unique.append(document)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="only chunk and print counts")
    parser.add_argument("--supabase", action="store_true", help="also write the chunks to Supabase")
    args = parser.parse_args()

    from src.rag import embeddings, faiss_store
    from src.rag.chunking import chunk_document

    documents = load_corpus_documents()
    chunks = []
    for document in documents:
        document_chunks = chunk_document(document)
        print(f"{document.doc_id}: {len(document_chunks)} chunks")
        chunks.extend(document_chunks)
    print(f"Total: {len(chunks)} chunks")

    if args.dry_run:
        return 0

    vectors = embeddings.embed_texts([chunk.content for chunk in chunks])
    source_files = sorted(path for path in CORPUS_DIR.iterdir() if path.suffix.lower() in {".pdf", ".csv"})
    run_dir = faiss_store.save_run(FAISS_DIR, chunks, vectors, embeddings.model_name(), source_files)
    print(f"Saved FAISS run: {run_dir}")

    if args.supabase:
        from src.rag.ingest import ingest_documents
        from src.services.fetch_postgre_database import get_connection

        with get_connection() as conn:
            result = ingest_documents(conn, documents)
        print(f"Supabase: {result['chunks']} chunks, {result['embedded']} newly embedded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
