"""GET /evacuation-centers: search the evacuation centers by city and barangay."""

from fastapi import APIRouter, Depends, Query

from src.api.deps import database_unavailable, get_db, resolve_city
from src.api.schemas import PLAIN_NAME_PATTERN, EvacuationCenterOut
from src.services.fetch_postgre_database import fetch_evacuation_centers

router = APIRouter(tags=["evacuation"])


@router.get("/evacuation-centers", response_model=list[EvacuationCenterOut])
def list_evacuation_centers(
    city: str | None = Query(default=None, max_length=60, description='A Metro Manila city, e.g. "Parañaque".'),
    barangay: str | None = Query(default=None, max_length=80, pattern=PLAIN_NAME_PATTERN),
    limit: int = Query(default=20, ge=1, le=50),
    get_connection=Depends(get_db),
) -> list[EvacuationCenterOut]:
    """List evacuation centers. For the nearest *safe* center to a person, use POST /chat."""
    city_key = resolve_city(city, None) if city else None
    try:
        rows = fetch_evacuation_centers(get_connection(), city=city_key, barangay=barangay, limit=limit)
    except Exception as error:
        raise database_unavailable(error)
    return [EvacuationCenterOut(**row) for row in rows]
