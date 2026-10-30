"""POST /requests: submit a supply or assistance request.

There is deliberately no public way to read requests back: they can hold a contact phone
number. Barangay and DRRMO staff read them from the database until an access-controlled
staff view exists.
"""

from fastapi import APIRouter, Depends

from src.api.deps import database_unavailable, get_db, limit_requests, resolve_city
from src.api.schemas import Received, RequestCreate
from src.services.fetch_postgre_database import insert_request

router = APIRouter(tags=["requests"])

RECEIVED_MESSAGE = (
    "Your request was received and saved as unverified. It does not mean help is on the way: "
    "if you are in danger, call 911 and contact your barangay or local DRRMO.\n"
    "Natanggap ang inyong request at naitala bilang hindi pa na-verify. Hindi ito nangangahulugang paparating na ang tulong: "
    "kung nasa panganib, tumawag sa 911 at makipag-ugnayan sa barangay o lokal na DRRMO."
)


@router.post("/requests", status_code=201, response_model=Received, dependencies=[Depends(limit_requests("requests"))])
def create_request(body: RequestCreate, get_connection=Depends(get_db)) -> Received:
    """Save a supply or assistance request. It is a record for staff to review, not a dispatch."""
    city = resolve_city(body.city, body.location)
    try:
        insert_request(
            get_connection(),
            kind=body.kind,
            city=city,
            barangay=body.barangay,
            details=body.details,
            people_count=body.people_count,
            contact_phone=body.contact_phone,
            lat=body.location.lat if body.location else None,
            lng=body.location.lng if body.location else None,
        )
    except Exception as error:
        raise database_unavailable(error)
    return Received(status="unverified", message=RECEIVED_MESSAGE)
