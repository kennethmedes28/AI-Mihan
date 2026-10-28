"""Save and load embeddings as a FAISS index on disk.

Every run writes its own folder, so an earlier run is never overwritten:

    faiss_vectors/<YYYYmmdd-HHMMSS>_<model name>/
        index.faiss     the vectors (exact cosine search; vectors are normalized)
        chunks.json     the text and metadata of each chunk, in the same order as the index
        metadata.json   how the run was made (model, vector size, time, source files)

Row ``i`` of the index belongs to ``chunks.json[i]``.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

from src.rag.models import Chunk

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
METADATA_FILE = "metadata.json"


def file_sha256(path: Path) -> str:
    """Fingerprint of a source file, so a run records exactly what it was built from."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run_name(model_name: str) -> str:
    """A name that is different for every run, e.g. ``20261028-194500_multilingual-e5-large``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{model_name.split('/')[-1]}"


def save_run(
    base_dir: Path,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    model_name: str,
    source_files: list[Path],
) -> Path:
    """Write one run to a new folder under ``base_dir`` and return that folder."""
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk needs exactly one embedding.")

    vectors = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product = cosine for normalized vectors
    index.add(vectors)

    run_dir = base_dir / make_run_name(model_name)
    run_dir.mkdir(parents=True, exist_ok=False)  # fails rather than overwrite

    faiss.write_index(index, str(run_dir / INDEX_FILE))

    chunk_records = [
        {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "chunk_idx": chunk.chunk_idx,
            "content": chunk.content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    (run_dir / CHUNKS_FILE).write_text(json.dumps(chunk_records, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "run_name": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "embedding_model": model_name,
        "embedding_dim": int(vectors.shape[1]),
        "index_type": "IndexFlatIP (cosine, normalized vectors)",
        "chunk_count": len(chunks),
        "source_files": [{"name": path.name, "sha256": file_sha256(path)} for path in source_files],
    }
    (run_dir / METADATA_FILE).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return run_dir


def load_run(run_dir: Path) -> tuple[faiss.Index, list[dict], dict]:
    """Read a saved run back: ``(index, chunk records, metadata)``."""
    index = faiss.read_index(str(run_dir / INDEX_FILE))
    chunk_records = json.loads((run_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / METADATA_FILE).read_text(encoding="utf-8"))
    return index, chunk_records, metadata
