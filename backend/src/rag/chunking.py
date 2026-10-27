"""Split documents into chunks (see "Chunking" in docs/ingestion_details.md).

Different document types are split differently:

    advisory  short bulletins: kept whole if short, otherwise split by section
    manual    long guides: split by heading, then by size, heading path added
    record    one chunk per line (evacuation centers, hotlines)
    report    one chunk for the whole text (a community report)

Sizes are counted in words, as a simple stand-in for tokens (one word is
roughly 1.3 tokens). Every chunk starts with a one-line header such as
``[PAGASA | advisory | 2026-10-24 | Manila | typhoon]`` so it makes sense on its own.
"""

import re

from src.rag.models import Chunk, Document

# Words per chunk and words repeated between neighbouring chunks.
CHUNK_SETTINGS = {
    "advisory": {"max_words": 300, "overlap_words": 30},
    "manual": {"max_words": 400, "overlap_words": 50},
}
WHOLE_ADVISORY_WORDS = 450  # an advisory this short stays in one chunk

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


def build_header(document: Document) -> str:
    """Write the one-line header from the document's metadata."""
    metadata = document.metadata
    parts = [
        metadata.get("agency"),
        document.doc_type,
        (metadata.get("issued_at") or "")[:10],  # date only
        metadata.get("city"),
        metadata.get("hazard"),
    ]
    parts = [str(part) for part in parts if part]
    return f"[{' | '.join(parts)}]" if parts else ""


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown-style text into ``(heading path, body)`` sections.

    The heading path joins the headings above a section, for example
    "Typhoons > Before a typhoon". Text without headings is one section.
    """
    sections = []
    path: list[str] = []  # current heading at each level
    body: list[str] = []

    def finish_section():
        text_body = "\n".join(body).strip()
        if text_body:
            sections.append((" > ".join(path), text_body))
        body.clear()

    for line in text.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            finish_section()
            level = len(match.group(1))
            del path[level - 1 :]  # drop this level and anything deeper
            path.append(match.group(2).strip())
        else:
            body.append(line)
    finish_section()

    return sections


def split_into_pieces(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Pack paragraphs into pieces of at most ``max_words`` words.

    Each new piece starts with the last ``overlap_words`` words of the
    previous one, so a sentence cut at the boundary appears in both.
    Line breaks inside a piece become spaces.
    """
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    pieces = []
    current: list[str] = []

    for paragraph in re.split(r"\n\s*\n", text):
        words = paragraph.split()
        if not words:
            continue

        # This paragraph does not fit: close the current piece first.
        if current and len(current) + len(words) > max_words:
            pieces.append(" ".join(current))
            current = current[-overlap_words:] if overlap_words else []

        current.extend(words)

        # A single paragraph longer than max_words is cut into windows.
        while len(current) > max_words:
            pieces.append(" ".join(current[:max_words]))
            current = current[max_words - overlap_words :]

    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_document(document: Document) -> list[Chunk]:
    """Split one document into chunks, numbered from 0.

    Raises:
        ValueError: if the document type is not one of the known types.
    """
    header = build_header(document)
    metadata = {**document.metadata, "doc_type": document.doc_type}

    if document.doc_type == "record":
        bodies = [("", line.strip()) for line in document.text.splitlines() if line.strip()]
    elif document.doc_type == "report":
        bodies = [("", document.text.strip())]
    elif document.doc_type in CHUNK_SETTINGS:
        bodies = _split_long_document(document)
    else:
        raise ValueError(f"Unknown doc_type: {document.doc_type!r}")

    chunks = []
    for heading_path, body in bodies:
        if not body:
            continue
        content = "\n".join(part for part in (header, heading_path, body) if part)
        chunks.append(Chunk(document.doc_id, len(chunks), content, dict(metadata)))
    return chunks


def _split_long_document(document: Document) -> list[tuple[str, str]]:
    """Return ``(heading path, text)`` pieces for advisories and manuals."""
    settings = CHUNK_SETTINGS[document.doc_type]

    if document.doc_type == "advisory" and len(document.text.split()) <= WHOLE_ADVISORY_WORDS:
        return [("", document.text.strip())]

    pieces = []
    for heading_path, section_text in split_sections(document.text):
        for piece in split_into_pieces(section_text, settings["max_words"], settings["overlap_words"]):
            pieces.append((heading_path, piece))
    return pieces
