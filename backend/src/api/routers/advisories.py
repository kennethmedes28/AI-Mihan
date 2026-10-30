"""GET /advisories/latest: the current official PAGASA bulletins.

Nothing is invented. When the pages cannot be read, or look like an error or block page,
the answer says so and points to the official sites instead of returning empty or stale text.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from agents.backup_sources import is_pagasa_reliable
from agents.fetch_weather_using_browser import fetch_weather_pages
from agents.location_agent import find_city, plain_text
from src.api.deps import UNKNOWN_CITY
from src.api.schemas import AdvisoriesOut, AdvisoryOut
from src.rag.models import PHILIPPINE_TZ

logger = logging.getLogger(__name__)

router = APIRouter(tags=["advisories"])

EXCERPT_CHARS = 1200
ADVISORY_QUESTION = "flood rainfall warning advisory"
UNAVAILABLE_NOTE = (
    "The official bulletins could not be read right now. Check pagasa.dost.gov.ph and ndrrmc.gov.ph, "
    "or contact your barangay or local DRRMO.\n"
    "Hindi mabasa ang opisyal na bulletin ngayon. Tingnan ang pagasa.dost.gov.ph at ndrrmc.gov.ph, "
    "o makipag-ugnayan sa barangay o lokal na DRRMO."
)
AVAILABLE_NOTE = "Shortened text from the official pages. Read the full bulletin at pagasa.dost.gov.ph."


@router.get("/advisories/latest", response_model=AdvisoriesOut)
def latest_advisories(city: str | None = Query(default=None, max_length=60)) -> AdvisoriesOut:
    """Return the current PAGASA flood and rainfall pages.

    ``city`` only marks which pages mention it (``mentions_city``); the bulletins are national.
    """
    city_key = None
    if city:
        city_key = find_city(city)
        if city_key is None:
            raise HTTPException(status_code=422, detail=UNKNOWN_CITY)

    now = datetime.now(PHILIPPINE_TZ)
    try:
        pages = fetch_weather_pages(ADVISORY_QUESTION)
    except Exception as error:  # PAGASA down or blocked: say so, do not fail.
        logger.warning("PAGASA pages unavailable: %s", error)
        pages = []

    if not is_pagasa_reliable(pages):
        return AdvisoriesOut(available=False, fetched_at=now, advisories=[], note=UNAVAILABLE_NOTE)

    advisories = [
        AdvisoryOut(
            title=page["title"] or page["source"],
            source=page["source"],
            excerpt=page["text"][:EXCERPT_CHARS],
            mentions_city=(city_key in plain_text(page["text"])) if city_key else None,
        )
        for page in pages
        if page.get("text", "").strip()
    ]
    return AdvisoriesOut(available=True, fetched_at=now, advisories=advisories, note=AVAILABLE_NOTE)
