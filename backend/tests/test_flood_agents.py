"""Tests for the flood agents. No network or database is used.

Run from the backend folder:  python -m unittest discover -s tests -v
"""

import sys
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("bm25s", types.ModuleType("bm25s"))  # optional search library, not needed here

from agents import contacts_agent, evacuation_agent, flood_risk_agent, location_agent, route_agent
from agents.flood_types import (
    RISK_CRITICAL, RISK_HIGH, RISK_LOW, RISK_WATCH, ChatContext, FloodZone, Location,
)
from langgraph_orchestrator.intents import (
    CONTACTS, EMERGENCY, EVACUATION, FLOOD_RISK, REPORT, ROUTE, WEATHER, classify_intent,
)
from src.rag.models import PHILIPPINE_TZ
from utils.geo import circle_polygon, haversine_m, point_to_segment_m

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=PHILIPPINE_TZ)
MARIKINA = Location("Marikina", 14.6507, 121.1029, "marikina")


def rain_data(next_hour_mm: float, past_mm_per_hour: float = 0.0) -> dict:
    """Fake Open-Meteo rain: 72 past hours at one rate, then 24 hours with the same rain each hour."""
    times, amounts = [], []
    for offset in range(-72, 25):
        hour = datetime(2026, 10, 28, 12) + (datetime(2026, 10, 29) - datetime(2026, 10, 28)) * offset / 24
        times.append(hour.strftime("%Y-%m-%dT%H:%M"))
        amounts.append(past_mm_per_hour if offset <= 0 else next_hour_mm)
    return {"hourly": {"time": times, "precipitation": amounts}}


class GeoTests(unittest.TestCase):
    def test_haversine_one_degree_of_latitude(self):
        self.assertAlmostEqual(haversine_m(14.0, 121.0, 15.0, 121.0), 111_195, delta=300)

    def test_point_to_segment_distance(self):
        # Point 111 m north of the middle of an east-west segment.
        distance = point_to_segment_m(14.001, 121.0, 14.0, 120.99, 14.0, 121.01)
        self.assertAlmostEqual(distance, 111, delta=2)

    def test_circle_polygon_is_closed(self):
        ring = circle_polygon(14.6, 121.0, 100)
        self.assertEqual(ring[0], ring[-1])


class IntentTests(unittest.TestCase):
    def test_intents(self):
        cases = {
            "Saan ang pinakamalapit na evacuation center?": EVACUATION,
            "paano pumunta doon?": ROUTE,
            "I want to report a flooded street": REPORT,
            "hotline ng Marikina": CONTACTS,
            "Babaha ba sa Pasig bukas?": FLOOD_RISK,
            "Uulan ba bukas?": WEATHER,
            "tulong, nalulunod kami": EMERGENCY,
        }
        for message, expected in cases.items():
            self.assertEqual(classify_intent(message), expected, message)


class LocationTests(unittest.TestCase):
    def test_city_only(self):
        found = location_agent.resolve_location("Nasa Parañaque ako")
        self.assertEqual((found.city, found.precision), ("paranaque", "city"))

    def test_coordinates(self):
        found = location_agent.resolve_location("14.6507, 121.1029")
        self.assertEqual((found.city, found.precision), ("marikina", "gps"))

    def test_barangay_uses_geocoder(self):
        found = location_agent.resolve_location(
            "Brgy. Nangka, Marikina", geocoder=lambda query: (14.6700, 121.1100)
        )
        self.assertEqual((found.barangay, found.precision), ("Nangka", "barangay"))
        self.assertEqual(found.lat, 14.67)

    def test_barangay_falls_back_to_city(self):
        found = location_agent.resolve_location("Brgy. Nangka, Marikina", geocoder=lambda query: None)
        self.assertEqual(found.precision, "city")

    def test_no_place(self):
        self.assertIsNone(location_agent.resolve_location("hello there"))

    def test_coordinates_outside_ncr_ignored(self):
        self.assertIsNone(location_agent.find_coordinates("10.3157, 123.8854"))


class FloodRiskTests(unittest.TestCase):
    def assess(self, rain, pagasa=""):
        return flood_risk_agent.assess_flood_risk("marikina", pagasa, now=NOW, rain=rain, river={"daily": {"river_discharge": [1.0]}})

    def test_dry_is_low(self):
        self.assertEqual(self.assess(rain_data(0.0)).level, RISK_LOW)

    def test_heavy_hourly_rain_is_critical(self):
        self.assertEqual(self.assess(rain_data(35.0)).level, RISK_CRITICAL)

    def test_moderate_rain_is_watch_or_higher(self):
        # 8 mm/h for 24 h = 192 mm total -> the daily total makes this critical; hourly alone is only watch.
        self.assertIn(self.assess(rain_data(8.0)).level, (RISK_HIGH, RISK_CRITICAL))
        self.assertEqual(self.assess(rain_data(1.8)).level, RISK_WATCH)  # 43 mm in 24 h

    def test_pagasa_red_warning(self):
        self.assertEqual(self.assess(rain_data(0.0), "Red Rainfall Warning over Metro Manila").level, RISK_CRITICAL)

    def test_soaked_ground_raises_a_dry_forecast_to_watch(self):
        self.assertEqual(self.assess(rain_data(0.0, past_mm_per_hour=2.0)).level, RISK_WATCH)

    def test_missing_data_is_reported(self):
        with mock.patch.object(flood_risk_agent, "download_rain", side_effect=RuntimeError("down")), \
             mock.patch.object(flood_risk_agent, "download_river", side_effect=RuntimeError("down")):
            risk = flood_risk_agent.assess_flood_risk("marikina", now=NOW)
        self.assertEqual(set(risk.missing), {"rain forecast", "river flow"})


class EvacuationTests(unittest.TestCase):
    ROWS = [
        {"name": "Far School", "lat": 14.70, "lng": 121.10, "city": "Marikina"},
        {"name": "Near Gym", "lat": 14.652, "lng": 121.103, "city": "Marikina"},
        {"name": "Flooded Hall", "lat": 14.6508, "lng": 121.1030, "city": "Marikina"},
        {"name": "No Position", "lat": None, "lng": None, "city": "Marikina"},
    ]

    def test_ranks_by_distance_and_drops_flooded(self):
        zone = FloodZone(14.6508, 121.1030, 100, "report")
        centers, flooded = evacuation_agent.rank_centers(MARIKINA, self.ROWS, [zone])
        self.assertEqual([c["name"] for c in centers], ["Near Gym", "Far School"])
        self.assertEqual(flooded, 1)  # only "Flooded Hall" is inside the 100 m circle

    def test_answer_sets_destination(self):
        context = ChatContext(location=MARIKINA, get_connection=lambda: object(), now=NOW)
        with mock.patch.object(evacuation_agent, "fetch_evacuation_centers", return_value=self.ROWS), \
             mock.patch.object(evacuation_agent, "assess_flood_risk", return_value=mock.Mock(level=RISK_LOW)), \
             mock.patch.object(evacuation_agent, "collect_zones", return_value=[FloodZone(14.6508, 121.1030, 100, "report")]):
            reply = evacuation_agent.answer_evacuation_question("nearest evacuation center?", context)
        self.assertIn("Near Gym", reply.text)
        self.assertEqual(reply.destination.name, "Near Gym")

    def test_asks_for_location(self):
        reply = evacuation_agent.answer_evacuation_question("nearest evacuation center?", ChatContext())
        self.assertEqual(reply.pending_intent, "evacuation")


class RouteTests(unittest.TestCase):
    # A straight street going north from (14.60, 121.00); [lng, lat] order.
    DIRECT = route_agent.Route(1000, 120, [[121.0, 14.600], [121.0, 14.605], [121.0, 14.610]])
    DETOUR = route_agent.Route(1500, 200, [[121.0, 14.600], [121.01, 14.605], [121.0, 14.610]])

    def test_picks_the_clear_route(self):
        zone = FloodZone(14.605, 121.0, 100, "report")
        self.assertIs(route_agent.pick_safe_route([self.DIRECT, self.DETOUR], [zone]), self.DETOUR)

    def test_shortest_when_nothing_flooded(self):
        self.assertIs(route_agent.pick_safe_route([self.DETOUR, self.DIRECT], []), self.DIRECT)

    def test_none_when_every_route_hits_a_zone(self):
        zones = [FloodZone(14.605, 121.0, 100, "report"), FloodZone(14.605, 121.01, 100, "report")]
        self.assertIsNone(route_agent.pick_safe_route([self.DIRECT, self.DETOUR], zones))

    def test_trip_phrases(self):
        self.assertEqual(route_agent.parse_trip("route from Brgy. Nangka to Marikina Sports Center"),
                         ("Brgy. Nangka", "Marikina Sports Center"))

    def test_zone_containing_start_is_not_sent_to_engine(self):
        start = Location("Here", 14.605, 121.0, "manila")
        end = Location("There", 14.62, 121.0, "manila")
        zone = FloodZone(14.605, 121.0, 100, "report")
        self.assertEqual(route_agent.zones_to_avoid(start, end, [zone]), [])

    def test_answer_without_api_key_gives_fallback(self):
        context = ChatContext(location=MARIKINA, destination=Location("Gym", 14.66, 121.11, "marikina"), now=NOW)
        with mock.patch.object(route_agent, "assess_flood_risk", return_value=mock.Mock(level=RISK_LOW)), \
             mock.patch.dict("os.environ", {}, clear=True):
            reply = route_agent.answer_route_question("how do I get there?", context)
        self.assertIn("cannot calculate a route", reply.text)


class ContactsTests(unittest.TestCase):
    def test_city_numbers_come_from_the_file(self):
        text = contacts_agent.format_contacts("marikina")
        self.assertIn("161", text)
        self.assertIn("911", text)

    def test_accented_city_matches(self):
        self.assertIn("(02) 8829-0922", contacts_agent.format_contacts("paranaque"))


if __name__ == "__main__":
    unittest.main()
