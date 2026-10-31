"""Staff endpoints: watch and review the reports and requests that residents submit.

Everything here needs the staff key (``Authorization: Bearer <STAFF_API_KEY>``). Unlike the public
endpoints it shows the full text, the exact position and, for requests, the contact phone number,
so the router is only registered when ``STAFF_API_KEY`` is set.

There is one shared key and no user accounts, so a status change records what changed but not who
changed it. Per-person logins are the next step if more than a few trusted people use this.
"""

import logging
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import database_unavailable, get_db, limit_requests, require_staff, resolve_city
from src.api.schemas import (
    ReportStatus,
    ReportStatusUpdate,
    RequestStatus,
    RequestStatusUpdate,
    StaffReportOut,
    StaffRequestOut,
    StaffSummary,
    StatusChanged,
    TableSummary,
)
from src.services.fetch_postgre_database import (
    fetch_reports_for_staff,
    fetch_requests_for_staff,
    summarize_table,
    update_status,
)

logger = logging.getLogger(__name__)

# The rate limit runs first, so wrong keys are slowed down before they are even checked.
router = APIRouter(
    prefix="/staff",
    tags=["staff"],
    dependencies=[Depends(limit_requests("staff")), Depends(require_staff)],
)

NOT_FOUND = "No item with that id."


@router.get("/summary", response_model=StaffSummary)
def summary(get_connection=Depends(get_db)) -> StaffSummary:
    """Counts by status and city for reports and requests, and how long the oldest unreviewed item has waited."""
    try:
        connection = get_connection()
        return StaffSummary(
            reports=TableSummary(**summarize_table(connection, "reports")),
            requests=TableSummary(**summarize_table(connection, "requests")),
        )
    except Exception as error:
        raise database_unavailable(error)


@router.get("/reports", response_model=list[StaffReportOut])
def list_reports(
    status: ReportStatus | None = None,
    city: str | None = Query(default=None, max_length=60),
    hazard: Literal["flood", "landslide", "other"] | None = None,
    limit: int = Query(default=20, ge=1, le=50),
    get_connection=Depends(get_db),
) -> list[StaffReportOut]:
    """The newest reports, unredacted. Filter with ``status=unverified`` to see what still needs review."""
    city_key = resolve_city(city, None) if city else None
    try:
        rows = fetch_reports_for_staff(get_connection(), status=status, city=city_key, hazard=hazard, limit=limit)
    except Exception as error:
        raise database_unavailable(error)
    return [StaffReportOut(**row) for row in rows]


@router.get("/requests", response_model=list[StaffRequestOut])
def list_requests(
    status: RequestStatus | None = None,
    city: str | None = Query(default=None, max_length=60),
    kind: Literal["supply", "assistance"] | None = None,
    limit: int = Query(default=20, ge=1, le=50),
    get_connection=Depends(get_db),
) -> list[StaffRequestOut]:
    """The newest requests, with contact numbers. Filter with ``status=unverified`` to see what still needs review."""
    city_key = resolve_city(city, None) if city else None
    try:
        rows = fetch_requests_for_staff(get_connection(), status=status, city=city_key, kind=kind, limit=limit)
    except Exception as error:
        raise database_unavailable(error)
    return [StaffRequestOut(**row) for row in rows]


@router.patch("/reports/{report_id}", response_model=StatusChanged)
def set_report_status(report_id: UUID, body: ReportStatusUpdate, get_connection=Depends(get_db)) -> StatusChanged:
    """Mark a report verified or rejected. Rejected reports are dropped from the public list, routing and the evacuation filter; verified and unverified ones are treated alike."""
    return change_status("reports", report_id, body.status, get_connection)


@router.patch("/requests/{request_id}", response_model=StatusChanged)
def set_request_status(request_id: UUID, body: RequestStatusUpdate, get_connection=Depends(get_db)) -> StatusChanged:
    """Move a request along: verified, assigned (someone is handling it) or closed."""
    return change_status("requests", request_id, body.status, get_connection)


def change_status(table: str, row_id: UUID, status: str, get_connection) -> StatusChanged:
    """Update one row and say if it existed. Only the id and new status are logged, never the text."""
    try:
        found = update_status(get_connection(), table, row_id, status)
    except Exception as error:
        raise database_unavailable(error)
    if not found:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    logger.info("Staff set %s %s to %s", table, row_id, status)
    return StatusChanged(id=row_id, status=status)
