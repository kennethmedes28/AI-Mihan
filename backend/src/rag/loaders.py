"""Load source material as ``Document`` objects, ready for chunking.

Sources:
    * text, Markdown and PDF files (preparedness manuals)
    * a CSV of evacuation centers
    * the official PAGASA pages (reuses the fetcher in agents/)

PDF reading needs the optional ``pypdf`` library: ``uv sync --extra rag``.
"""

import csv
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from agents.fetch_weather_using_browser import PAGASA_PAGES, load_clean_page
from src.rag.models import PHILIPPINE_TZ, Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
PAGASA_VALID_HOURS = 24  # a bulletin is treated as current for a day


def read_file_text(path: Path) -> str:
    """Return the text of a .txt, .md or .pdf file."""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("pypdf is not installed. Run: uv sync --extra rag") from error
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)

    return path.read_text(encoding="utf-8")


def load_file(path: str | Path, doc_type: str = "manual", metadata: dict | None = None) -> Document:
    """Load one file as a document. The file name (without extension) is its id."""
    path = Path(path)
    return Document(
        doc_id=path.stem,
        text=read_file_text(path),
        doc_type=doc_type,
        metadata={"source": path.name, **(metadata or {})},
    )


def load_directory(directory: str | Path, doc_type: str = "manual", metadata: dict | None = None) -> list[Document]:
    """Load every supported file under a folder. Unreadable files are skipped and logged."""
    documents = []
    for path in sorted(Path(directory).rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            documents.append(load_file(path, doc_type, metadata))
        except Exception as error:
            logger.error("Could not load %s: %s", path, error)
    return documents


def load_evacuation_centers_csv(path: str | Path) -> list[Document]:
    """Load evacuation centers from a CSV with columns: name, barangay, city, capacity.

    One document per city, with one line per center, so each center becomes
    one chunk and searches can be limited to a city.
    """
    lines_by_city = defaultdict(list)

    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            line = f"{row['name']}, Barangay {row['barangay']}, {row['city']}, capacity {row['capacity']}"
            lines_by_city[row["city"].strip()].append(line)

    return [
        Document(
            doc_id=f"evacuation-centers-{city.lower().replace(' ', '-')}",
            text="\n".join(lines),
            doc_type="record",
            metadata={"city": city, "agency": "LGU", "trust_tier": "official", "hazard": None},
        )
        for city, lines in lines_by_city.items()
    ]


def load_pagasa_documents() -> list[Document]:
    """Load the official PAGASA pages as advisories that expire after a day.

    Pages that cannot be downloaded are skipped and logged.
    """
    fetched_at = datetime.now(PHILIPPINE_TZ)
    valid_until = (fetched_at + timedelta(hours=PAGASA_VALID_HOURS)).isoformat()

    documents = []
    for key, page in PAGASA_PAGES.items():
        try:
            title, text = load_clean_page(page["url"])
        except Exception as error:
            logger.error("Could not load PAGASA page %s: %s", key, error)
            continue

        documents.append(
            Document(
                doc_id=f"pagasa-{key}",
                text=text,
                doc_type="advisory",
                metadata={
                    "source": page["url"],
                    "title": title,
                    "agency": "PAGASA",
                    "trust_tier": "official",
                    "issued_at": fetched_at.isoformat(),
                    "valid_until": valid_until,
                },
            )
        )
    return documents
