"""Collects the map areas that routes and evacuation advice must avoid.

Two sources, both drawn as circles:

    * recent community flood reports that have a position (``source="report"``)
    * known flood-prone spots from ``data/flood_prone_areas.csv`` (``source="mapped"``),
      used only while the city's flood risk is high or critical

Reports are unverified, but they are used anyway: a wrong report only costs a
detour, while ignoring a true one can put someone in floodwater.
"""

import csv
import logging
from pathlib import Path

from agents.flood_types import RISK_HIGH, RISK_ORDER, ChatContext, FloodZone
from src.services.fetch_postgre_database import fetch_report_points

logger = logging.getLogger(__name__)

MAPPED_AREAS_FILE = Path(__file__).resolve().parents[1] / "data" / "flood_prone_areas.csv"
REPORT_VALID_HOURS = 6  # a street flood report is treated as current for this long
REPORT_ZONE_RADIUS_M = 150
DEFAULT_MAPPED_RADIUS_M = 300


def load_mapped_areas(city: str | None = None, path: Path = MAPPED_AREAS_FILE) -> list[FloodZone]:
    """Read the known flood-prone spots, optionally only those in one city.

    CSV columns: name, city, lat, lng, radius_m, note. Rows with bad numbers are skipped.
    """
    if not path.exists():
        return []

    zones = []
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if city and row.get("city", "").strip().lower() != city.lower():
                continue
            try:
                zones.append(
                    FloodZone(
                        lat=float(row["lat"]),
                        lng=float(row["lng"]),
                        radius_m=float(row.get("radius_m") or DEFAULT_MAPPED_RADIUS_M),
                        source="mapped",
                        note=row.get("name", ""),
                    )
                )
            except (KeyError, ValueError):
                logger.warning("Skipping bad row in %s: %s", path, row)
    return zones


def report_zones(context: ChatContext) -> list[FloodZone]:
    """Zones from recent community reports anywhere in Metro Manila (a route can cross cities).

    Returns none if there is no database.
    """
    if context.get_connection is None:
        return []
    try:
        rows = fetch_report_points(context.get_connection(), hours=REPORT_VALID_HOURS)
    except Exception as error:  # Missing lat/lng columns, database down, and so on.
        logger.warning("Could not read flood reports: %s", error)
        return []

    return [
        FloodZone(
            lat=row["lat"],
            lng=row["lng"],
            radius_m=REPORT_ZONE_RADIUS_M,
            source="report",
            note=(row.get("body") or "")[:80],
        )
        for row in rows
    ]


def collect_zones(context: ChatContext, city: str | None, risk_level: str | None) -> list[FloodZone]:
    """All areas to avoid right now for a city at the given flood risk level."""
    zones = report_zones(context)
    if risk_level and RISK_ORDER.index(risk_level) >= RISK_ORDER.index(RISK_HIGH):
        zones += load_mapped_areas(city)
    return zones
