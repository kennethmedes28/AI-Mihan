"""Load source material as ``Document`` objects, ready for chunking.

Sources:
    * text, Markdown and PDF files (preparedness manuals)
    * a CSV of evacuation centers a CSV of emergency hotlines and a CSV of NCR location elevations
    * the official PAGASA pages (reuses the fetcher in agents/)

PDF reading needs ``pypdf``. Scanned pages are read with the ``tesseract`` program.
"""

import csv
import logging
import re
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from agents.fetch_weather_using_browser import PAGASA_PAGES, load_clean_page
from src.rag.models import PHILIPPINE_TZ, Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
PAGASA_VALID_HOURS = 24  # a bulletin is treated as current for a day
MIN_PAGE_CHARS = 50  # a PDF page with less text than this is treated as a scan
OCR_DPI = 200


def ocr_pdf_page(path: Path, page_number: int) -> str:
    """Read one scanned PDF page with Tesseract (needs the ``tesseract`` program and PyMuPDF)."""
    import pymupdf  # imported here because it is only needed for scanned files

    with pymupdf.open(str(path)) as pdf, tempfile.TemporaryDirectory() as folder:
        image_path = Path(folder) / "page.png"
        pdf[page_number].get_pixmap(dpi=OCR_DPI).save(str(image_path))
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "-l", "eng"],
            capture_output=True, text=True, check=True,
        )
    return result.stdout


def read_pdf_text(path: Path) -> str:
    """Return the text of a PDF. Pages with no text layer (scans) are read with OCR."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("pypdf is not installed. Run: uv sync --extra rag") from error

    page_texts = []
    for number, page in enumerate(PdfReader(str(path)).pages):
        text = page.extract_text() or ""
        if len(text.strip()) < MIN_PAGE_CHARS:
            text = ocr_pdf_page(path, number)
        page_texts.append(text)
    return "\n\n".join(page_texts)


def read_file_text(path: Path) -> str:
    """Return the text of a .txt, .md or .pdf file."""
    if path.suffix.lower() == ".pdf":
        return read_pdf_text(path)

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


def load_emergency_contacts_csv(path: str | Path) -> list[Document]:
    """Load a hotline list (columns Agency, Category, Contact Type, Number, Notes).

    One line per number, so each number becomes one chunk. Rows still holding the
    "Add your own" placeholder are skipped because they have no real number.
    """
    lines = []

    # utf-8-sig drops the invisible marker Excel puts at the start of CSV files.
    with open(path, newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            number = row["Number"].strip()
            if not number or number.lower() == "add your own":
                continue
            line = f"{row['Agency']} ({row['Category']}), {row['Contact Type']}: {number}"
            if row["Notes"].strip():
                line += f". {row['Notes'].strip()}"
            lines.append(line)

    return [
        Document(
            doc_id=Path(path).stem,
            text="\n".join(lines),
            doc_type="record",
            metadata={"source": Path(path).name, "trust_tier": "official", "hazard": None},
        )
    ]


def load_ncr_locations_csv(path: str | Path) -> list[Document]:
    """Load sampled NCR points (columns location_id, latitude, longitude, elevation_m, coord_group, city).

    One document per city with one line per point, so each point becomes one chunk
    and searches can be limited to a city. Low elevation is useful flood context.
    """
    lines_by_city = defaultdict(list)

    with open(path, newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            city = re.sub(r"\s*\(.*\)$", "", row["city"].strip())  # "Pateros (municipality)" -> "Pateros"
            line = (
                f"{city} location {row['location_id']}: elevation {float(row['elevation_m']):g} m, "
                f"latitude {row['latitude']}, longitude {row['longitude']}"
            )
            lines_by_city[city].append(line)

    return [
        Document(
            doc_id=f"ncr-locations-{city.lower().replace(' ', '-')}",
            text="\n".join(lines),
            doc_type="record",
            metadata={"source": Path(path).name, "city": city, "trust_tier": "official", "hazard": None},
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
