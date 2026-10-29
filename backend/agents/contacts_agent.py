"""Contacts agent: looks up emergency numbers for the user's city from the contacts CSV.

No chat model is used: phone numbers are copied exactly from the file, never
generated, because a wrong digit can cost a life.
"""

import csv
import os
from functools import lru_cache
from pathlib import Path

from agents.flood_types import AgentReply, ChatContext
from agents.location_agent import ASK_FOR_LOCATION, plain_text, find_city
from utils.tracing import traced_agent

DEFAULT_CONTACTS = Path(__file__).resolve().parents[2] / "corpus" / "ph_emergency_contacts.csv"

# National lines shown to everyone, as (agency, number type).
NATIONAL_AGENCIES = ["National Emergency Hotline", "NDRRMC", "Philippine Red Cross", "MMDA"]
LOCAL_CATEGORIES = ["Local - City Rescue", "Local - Fire"]
MAX_PER_AGENCY = 3


@lru_cache(maxsize=1)
def load_contacts() -> list[dict]:
    """Read the contacts file once. The BOM at the start of the file is handled."""
    path = Path(os.getenv("CONTACTS_CSV", DEFAULT_CONTACTS))
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def contacts_for_city(city: str | None, contacts: list[dict] | None = None) -> list[tuple[str, list[str]]]:
    """Return ``[(agency, [numbers...])]``: the national lines, then the city's rescue and fire lines."""
    contacts = load_contacts() if contacts is None else contacts

    def numbers_by_agency(rows: list[dict]) -> list[tuple[str, list[str]]]:
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(row["Agency"], []).append(row["Number"])
        return [(agency, numbers[:MAX_PER_AGENCY]) for agency, numbers in grouped.items()]

    national = [row for row in contacts if row["Agency"] in NATIONAL_AGENCIES]
    local = []
    if city:
        local = [
            row
            for row in contacts
            if row["Category"] in LOCAL_CATEGORIES and city in plain_text(row["Agency"])
        ]
    return numbers_by_agency(national) + numbers_by_agency(local)


def format_contacts(city: str | None, contacts: list[dict] | None = None) -> str:
    """Write the numbers as a short list."""
    title = f"Emergency numbers for {city.title()}:" if city else "Emergency numbers:"
    lines = [title]
    lines += [f"- {agency}: {', '.join(numbers)}" for agency, numbers in contacts_for_city(city, contacts)]
    return "\n".join(lines)


@traced_agent("contacts_agent")
def answer_contacts_question(question: str, context: ChatContext) -> AgentReply:
    """List the hotlines for the city in the question or the user's saved city."""
    city = find_city(question) or (context.location.city if context.location else None)
    if city is None:
        return AgentReply(text=ASK_FOR_LOCATION, pending_intent="contacts")
    return AgentReply(text=format_contacts(city))
