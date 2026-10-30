"""Community reports: POST /reports to submit one, GET /reports to list the recent ones.

Reports are unverified community input. They never override PAGASA or NDRRMC information.
"""

from typing import Literal

from fastapi import APIRouter, Depends, Query

from src.api.deps import database_unavailable, get_db, limit_requests, resolve_city
from src.api.schemas import Received, ReportCreate, ReportOut
from src.services.fetch_postgre_database import fetch_recent_reports, insert_report
from utils.tracing import redact

router = APIRouter(tags=["reports"])

SAVED_MESSAGE = (
    "Thank you. Your report was saved as an unverified community report. If someone is in danger, call 911.\n"
    "Salamat. Naitala ang inyong ulat bilang hindi pa na-verify na ulat ng komunidad. Kung may nasa panganib, tumawag sa 911."
)


# These handlers are plain "def", not "async def": the database calls block, so FastAPI runs them in a worker thread.
@router.post("/reports", status_code=201, response_model=Received, dependencies=[Depends(limit_requests("reports"))])
def create_report(body: ReportCreate, get_connection=Depends(get_db)) -> Received:
    """Save a community report. A GPS point lets routes avoid the spot; a city alone is saved but cannot be mapped."""
    city = resolve_city(body.city, body.location)
    try:
        insert_report(
            get_connection(),
            hazard=body.hazard,
            city=city,
            barangay=body.barangay,
            body=body.body,
            lat=body.location.lat if body.location else None,
            lng=body.location.lng if body.location else None,
        )
    except Exception as error:
        raise database_unavailable(error)
    return Received(status="unverified", message=SAVED_MESSAGE)


@router.get("/reports", response_model=list[ReportOut])
def list_reports(
    city: str | None = Query(default=None, max_length=60),
    hazard: Literal["flood", "landslide", "other"] | None = None,
    limit: int = Query(default=10, ge=1, le=50),
    get_connection=Depends(get_db),
) -> list[ReportOut]:
    """List the newest reports, without rejected ones. Phone-number-like digits in the text are hidden."""
    city_key = resolve_city(city, None) if city else None
    try:
        rows = fetch_recent_reports(get_connection(), city=city_key, hazard=hazard, limit=limit)
    except Exception as error:
        raise database_unavailable(error)

    return [ReportOut(**{**row, "body": redact(row["body"])}) for row in rows if row["status"] != "rejected"]
