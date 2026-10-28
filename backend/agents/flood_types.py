"""Small data types shared by the flood agents.

Locations are stored in the chat state as plain dictionaries (``to_dict`` /
``from_dict``), so the conversation can be saved by the checkpointer.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable

# Flood risk levels, lowest first.
RISK_LOW = "low"
RISK_WATCH = "watch"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"
RISK_ORDER = [RISK_LOW, RISK_WATCH, RISK_HIGH, RISK_CRITICAL]

# How exactly a location is known, most exact first.
PRECISION_GPS = "gps"
PRECISION_BARANGAY = "barangay"
PRECISION_CITY = "city"


@dataclass
class Location:
    """A place the user is at or going to."""

    name: str
    lat: float
    lng: float
    city: str  # lower-case NCR city key, such as "quezon city"
    barangay: str | None = None
    precision: str = PRECISION_CITY

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Location | None":
        return cls(**data) if data else None


@dataclass
class FloodZone:
    """A circle on the map that people should not be sent through."""

    lat: float
    lng: float
    radius_m: float
    source: str  # "report" (community) or "mapped" (known flood-prone spot)
    note: str = ""


@dataclass
class ChatContext:
    """What an agent knows about the current conversation and its services."""

    location: Location | None = None
    destination: Location | None = None
    get_connection: Callable | None = None  # returns an open database connection
    now: datetime | None = None  # tests set this; normally the real time is used
    extra: dict = field(default_factory=dict)  # e.g. {"ors_api_key": "..."}


@dataclass
class AgentReply:
    """An agent's answer plus anything it wants remembered for the next message."""

    text: str
    location: Location | None = None  # new user location, if one was worked out
    destination: Location | None = None  # new destination, if one was chosen
    pending_intent: str | None = None  # set when the agent asked a question and wants the answer
