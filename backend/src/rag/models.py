"""Small data types shared by the RAG modules."""

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

PHILIPPINE_TZ = ZoneInfo("Asia/Manila")


@dataclass
class Chunk:
    """One searchable piece of a document.

    ``metadata`` follows docs/ingestion_details.md: hazard, city, issued_at,
    valid_until (ISO text), trust_tier, and so on. ``score`` is set by
    whichever stage last ranked the chunk (BM25, vector search, fusion, rerank).
    """

    doc_id: str
    chunk_idx: int
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0

    @property
    def id(self) -> str:
        """Unique id of the chunk, used to match the same chunk across searches."""
        return f"{self.doc_id}:{self.chunk_idx}"


@dataclass
class Document:
    """A whole source document, before it is split into chunks.

    ``doc_type`` decides how it is chunked: "advisory", "manual", "record"
    (one chunk per line) or "report" (one chunk in total).
    """

    doc_id: str
    text: str
    doc_type: str
    metadata: dict = field(default_factory=dict)


def parse_time(value: str | None) -> datetime | None:
    """Read an ISO timestamp. Times without a timezone are taken as Philippine time."""
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=PHILIPPINE_TZ)


@dataclass
class SearchFilter:
    """Limits applied before ranking (see "Metadata pre-filter" in the design doc).

    A chunk with no hazard or city in its metadata is general material
    (for example a preparedness manual), so it passes that part of the filter.
    """

    hazard: str | None = None
    city: str | None = None
    now: datetime | None = None  # chunks whose valid_until is before this are dropped

    def current_time(self) -> datetime:
        return self.now or datetime.now(PHILIPPINE_TZ)

    def matches(self, metadata: dict) -> bool:
        """Return True if a chunk with this metadata may be searched."""
        for name, wanted in (("hazard", self.hazard), ("city", self.city)):
            actual = metadata.get(name)
            if wanted and actual and actual.lower() != wanted.lower():
                return False

        valid_until = parse_time(metadata.get("valid_until"))
        return valid_until is None or valid_until >= self.current_time()
