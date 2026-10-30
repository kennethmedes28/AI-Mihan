"""Request and response bodies of the API (Pydantic models).

Validation lives here, so the routers stay short. Two rules apply everywhere:

    * unknown fields are rejected, so a typo in a field name is an error and not silently ignored;
    * every position must lie inside Metro Manila, which is all this service covers. A GPS
      point outside it would be matched to the nearest Metro Manila city and give a wrong answer.
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from agents.location_agent import NCR_BOUNDS

OUTSIDE_COVERAGE = (
    "This location is outside Metro Manila, which is all this service covers. "
    "If you are in danger, call 911 and contact your local DRRMO. "
    "Nasa labas ito ng Metro Manila, na siya lamang saklaw ng serbisyong ito. "
    "Kung nasa panganib, tumawag sa 911 at makipag-ugnayan sa inyong lokal na DRRMO."
)

# A short piece of text with the spaces around it removed.
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
# Letters (including the Spanish n with a tilde), digits, spaces, dots, apostrophes and hyphens.
# The query parameters use this pattern so "%" and "_" cannot widen a database search.
PLAIN_NAME_PATTERN = r"^[A-Za-zÑñ0-9 .'\-]+$"


class StrictModel(BaseModel):
    """Base class: unknown fields are an error."""

    model_config = ConfigDict(extra="forbid")


class Coordinates(StrictModel):
    """A GPS point inside Metro Manila."""

    lat: float
    lng: float

    @model_validator(mode="after")
    def inside_metro_manila(self) -> "Coordinates":
        west, south, east, north = NCR_BOUNDS
        # Written as "not inside" so that NaN, which fails every comparison, is rejected too.
        if not (south <= self.lat <= north and west <= self.lng <= east):
            raise ValueError(OUTSIDE_COVERAGE)
        return self


# --- Chat -----------------------------------------------------------------------------------


class ChatRequest(StrictModel):
    """One message from the user."""

    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    # Leave out to start a new conversation; send back the value from the first reply to continue it.
    # It must be a UUID so ids cannot be guessed from a pattern.
    thread_id: UUID | None = None
    location: Coordinates | None = None


class ChatResponse(BaseModel):
    """The assistant's reply."""

    thread_id: str
    reply: str
    # True when the reply is the fixed "I am not confident enough" text, so the app can style it differently.
    safe_answer: bool


# --- Places in reports and requests ----------------------------------------------------------


class PlaceFields(StrictModel):
    """Where something happened: a city name, a GPS point, or both."""

    city: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)] | None = None
    barangay: ShortText | None = None
    location: Coordinates | None = None

    @model_validator(mode="after")
    def needs_a_place(self):
        if self.city is None and self.location is None:
            raise ValueError("Give a city or a location.")
        return self


class ReportCreate(PlaceFields):
    """A community report of a hazard. Saved as unverified."""

    hazard: Literal["flood", "landslide", "other"] = "flood"
    body: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]


class ReportOut(BaseModel):
    """A report as listed publicly. Phone-number-like digits in the text are hidden."""

    hazard: str
    city: str
    barangay: str | None = None
    body: str
    status: str
    trust_level: str
    created_at: datetime


class RequestCreate(PlaceFields):
    """A supply or assistance request. Saved as unverified."""

    kind: Literal["supply", "assistance"]
    details: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]
    people_count: int | None = Field(default=None, ge=1, le=10000)
    # Optional. Digits with spaces, dashes, brackets and a leading plus.
    contact_phone: Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^\+?\d[\d \-()]{5,18}\d$")] | None = None


class Received(BaseModel):
    """Answer to a report or request that was saved."""

    status: str
    message: str


# --- Evacuation centers and advisories ------------------------------------------------------


class EvacuationCenterOut(BaseModel):
    """One evacuation center."""

    name: str
    city: str | None = None
    barangay: str | None = None
    type: str | None = None
    lat: float | None = None
    lng: float | None = None
    notes: str | None = None


class AdvisoryOut(BaseModel):
    """One official page (a PAGASA bulletin or advisory), shortened."""

    title: str
    source: str
    excerpt: str
    mentions_city: bool | None = None  # only set when a city was asked for


class AdvisoriesOut(BaseModel):
    """Current official advisories, or a plain statement that they could not be read."""

    available: bool
    fetched_at: datetime
    advisories: list[AdvisoryOut]
    note: str


# --- Health ---------------------------------------------------------------------------------


class HealthOut(BaseModel):
    """What works right now."""

    status: Literal["ok", "degraded"]
    checks: dict[str, str]
