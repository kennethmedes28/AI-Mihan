"""Tests for the answer gaps listed in the README: wider emergency phrases, "how long will it last?",
places outside Metro Manila, out-of-date PAGASA bulletins, source lists and kept barangays.
No network, database or model is used."""

import asyncio
import sys
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("bm25s", types.ModuleType("bm25s"))

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver

from agents import backup_sources, document_agents, flood_duration_agent, location_agent
from agents.coverage_agent import answer_outside_coverage
from agents.evacuation_agent import answer_evacuation_question
from agents.flood_types import AgentReply, ChatContext, Location
from langgraph_orchestrator import graph as chat_graph
from langgraph_orchestrator.intents import (
    DURATION, EMERGENCY, EVACUATION, FLOOD_RISK, REPORT, classify_intent, split_by_intent,
)
from src.rag.models import PHILIPPINE_TZ, Chunk

MANILA = Location("Divisoria, Manila", 14.60, 120.97, "manila", precision="place")
NOW = datetime(2026, 10, 31, 19, 0, tzinfo=PHILIPPINE_TZ)


class EmergencyPhraseTests(unittest.TestCase):
    def test_thigh_waist_and_chest_deep_are_emergencies(self):
        for message in [
            "Hanggang hita na ang baha. Saan ang pinakamalapit na evacuation center?",
            "Water is up to my waist. Where is the nearest evacuation center?",
            "the water is up to the chest",
            "waist-high na ang tubig",
            "thigh deep water outside",
            "hanggang balikat na ang tubig",
        ]:
            self.assertEqual(classify_intent(message), EMERGENCY, message)

    def test_knee_deep_is_still_not_an_emergency(self):
        self.assertEqual(classify_intent("baha hanggang tuhod, saan ang evacuation center"), EVACUATION)

    def test_a_dry_question_is_not_an_emergency(self):
        self.assertNotEqual(classify_intent("How long does the typhoon season last?"), EMERGENCY)


class DurationIntentTests(unittest.TestCase):
    def test_how_long_questions_are_duration_not_flood_risk(self):
        self.assertEqual(classify_intent("Gaano katagal ang baha sa Marikina?"), DURATION)
        self.assertEqual(classify_intent("How long will the flood last in Marikina?"), DURATION)
        self.assertEqual(classify_intent("Babaha ba sa Marikina ngayong gabi?"), FLOOD_RISK)

    def test_a_report_is_kept_when_the_user_also_asks_something_else(self):
        message = "I'm in Divisoria, Manila. I want to report a flood. Where can I go to an evacuation center, and how long will it last?"
        self.assertEqual([intent for intent, _ in split_by_intent(message)], [REPORT, EVACUATION, DURATION])

    def test_a_report_alone_is_not_split(self):
        self.assertEqual(split_by_intent("I want to report a flooded street in Marikina"), [])

    def test_report_inside_an_emergency_message_is_found(self):
        message = "Nasa Divisoria, Manila ako. Gusto kong mag-report ng baha hanggang baywang. Saan ako pwedeng pumunta na evacuation center at gaano katagal ang baha?"
        parts = split_by_intent(message, min_topics=1, in_emergency=True)
        self.assertEqual([intent for intent, _ in parts], [REPORT, EVACUATION, DURATION])
        self.assertEqual(split_by_intent(message), [])  # outside the emergency node an emergency is never split


class EvacuationSafetyTests(unittest.TestCase):
    def test_safety_steps_come_before_the_question_when_no_place_is_known(self):
        reply = answer_evacuation_question("Hanggang hita na ang baha. Saan ang pinakamalapit na evacuation center?", ChatContext())
        self.assertIn("911", reply.text)
        self.assertIn("Which city", reply.text)
        self.assertLess(reply.text.index("911"), reply.text.index("Which city"))
        self.assertEqual(reply.pending_intent, "evacuation")

    def test_a_calm_question_gets_only_the_question(self):
        reply = answer_evacuation_question("Saan ang pinakamalapit na evacuation center?", ChatContext())
        self.assertNotIn("911", reply.text)


class OutsideCoverageTests(unittest.TestCase):
    def test_places_outside_metro_manila_are_found(self):
        self.assertEqual(location_agent.find_outside_place("Nasa Tanza, Cavite ako. May data ka ba ng baha?"), "Tanza")
        self.assertEqual(location_agent.find_outside_place("I'm in Cavite"), "Cavite")

    def test_a_metro_manila_place_wins(self):
        self.assertIsNone(location_agent.find_outside_place("from Antipolo to Marikina, is the road safe?"))

    def test_streets_named_rizal_are_not_the_province(self):
        self.assertIsNone(location_agent.find_outside_place("Nasa Rizal Avenue, Manila ako"))

    def test_coordinates_outside_metro_manila(self):
        self.assertEqual(location_agent.find_outside_place("I am at 10.3157, 123.8854"), "those coordinates")

    def test_reply_says_not_covered_and_lists_national_numbers_without_mmda(self):
        reply = answer_outside_coverage("Nasa Tanza, Cavite ako. May data ka ba ng baha?", ChatContext())
        self.assertIn("cannot answer for Tanza", reply.text)
        self.assertIn("National Emergency Hotline: 911", reply.text)
        self.assertNotIn("MMDA", reply.text)
        self.assertNotIn("Which city", reply.text)


class RoutingTests(unittest.TestCase):
    def route(self, text, pending=None):
        message = chat_graph.add_messages([], [("user", text)])
        return chat_graph.route_message({"messages": message, "pending_intent": pending})

    def test_outside_place_goes_to_the_coverage_node(self):
        self.assertEqual(self.route("Nasa Tanza, Cavite ako. May data ka ba ng baha?"), chat_graph.OUTSIDE_NODE)
        self.assertEqual(self.route("Tanza", pending="evacuation"), chat_graph.OUTSIDE_NODE)

    def test_weather_outside_metro_manila_is_still_answered(self):
        self.assertEqual(self.route("Uulan ba bukas sa Antipolo?"), chat_graph.WEATHER_NODE)

    def test_emergency_outside_metro_manila_stays_an_emergency(self):
        self.assertEqual(self.route("Trapped kami sa Tanza, Cavite"), chat_graph.EMERGENCY_NODE)

    def test_duration_goes_to_its_own_node(self):
        self.assertEqual(self.route("Gaano katagal ang baha sa Marikina?"), chat_graph.DURATION_NODE)


class EmergencyNodeTests(unittest.TestCase):
    """The real graph with the agents replaced by fakes that record what they were asked."""

    def setUp(self):
        self.asked = {}

    def fakes(self):
        def evacuation(question, context):
            self.asked["evacuation"] = (question, dict(context.extra))
            return AgentReply(text="CENTERS", location=MANILA)

        def report(question, context):
            self.asked["report"] = question
            return AgentReply(text="REPORT SAVED", location=MANILA)

        def duration(question, context):
            self.asked["duration"] = question
            return AgentReply(text="DURATION ANSWER")

        return mock.patch.multiple(
            chat_graph,
            answer_evacuation_question=evacuation,
            answer_report_request=report,
            answer_duration_question=duration,
        )

    def ask(self, graph, text, thread="t"):
        return asyncio.run(chat_graph.ask_reply(graph, text, thread)).text

    def build(self):
        return chat_graph.build_graph(FakeListChatModel(responses=["x"]), checkpointer=MemorySaver())

    def test_one_message_gets_the_911_text_the_report_the_centers_and_the_duration(self):
        message = "Nasa Divisoria, Manila ako. Gusto kong mag-report ng baha hanggang baywang. Saan ako pwedeng pumunta na evacuation center at gaano katagal ang baha?"
        with self.fakes(), mock.patch.object(chat_graph, "resolve_location", return_value=MANILA):
            text = self.ask(self.build(), message)
        for part in ("call 911", "Emergency numbers for Manila", "REPORT SAVED", "CENTERS", "DURATION ANSWER"):
            self.assertIn(part, text)
        self.assertLess(text.index("call 911"), text.index("REPORT SAVED"))
        self.assertLess(text.index("REPORT SAVED"), text.index("CENTERS"))
        self.assertLess(text.index("CENTERS"), text.index("DURATION ANSWER"))
        self.assertTrue(self.asked["evacuation"][1]["safety_shown"])  # so the 911 text is not repeated

    def test_emergency_with_no_place_asks_for_it_after_the_911_text_and_then_lists_the_centers(self):
        with self.fakes(), mock.patch.object(chat_graph, "resolve_location", return_value=None):
            graph = self.build()
            first = self.ask(graph, "Water is up to my waist. Where is the nearest evacuation center?")
            self.assertLess(first.index("call 911"), first.index("Which city"))
            self.assertNotIn("CENTERS", first)

            with mock.patch.object(chat_graph, "resolve_location", return_value=MANILA):
                second = self.ask(graph, "Divisoria, Manila")
        self.assertIn("CENTERS", second)

    def test_a_plain_emergency_is_still_only_the_fixed_text(self):
        with self.fakes():
            text = self.ask(self.build(), "Trapped kami dito, tulong!")
        self.assertIn("call 911", text)
        self.assertNotIn("CENTERS", text)

    def test_emergency_outside_metro_manila_says_so_and_still_gives_911(self):
        text = self.ask(self.build(), "Trapped kami sa Tanza, Cavite")
        self.assertIn("call 911", text)
        self.assertIn("cannot answer for Tanza", text)

    def test_the_fixed_text_survives_a_failing_agent(self):
        with self.fakes(), mock.patch.object(chat_graph, "resolve_location", return_value=MANILA), \
             mock.patch.object(chat_graph, "answer_evacuation_question", side_effect=RuntimeError("boom")):
            text = self.ask(self.build(), "Water is up to my waist. Where is the nearest evacuation center?")
        self.assertIn("call 911", text)

    def test_report_with_another_question_saves_the_report_and_answers_the_rest(self):
        message = "I'm in Divisoria, Manila. I want to report a flood. Where can I go to an evacuation center, and how long will it last?"
        with self.fakes(), mock.patch.object(chat_graph, "resolve_location", return_value=MANILA):
            text = self.ask(self.build(), message)
        for part in ("REPORT SAVED", "CENTERS", "DURATION ANSWER"):
            self.assertIn(part, text)


class DurationAnswerTests(unittest.TestCase):
    def test_it_never_promises_a_time_and_says_so(self):
        text = flood_duration_agent.format_duration("marikina", [0.0] * 24, None, NOW)
        self.assertIn("cannot predict how long floodwater will stay", text)
        self.assertNotIn("LOW", text)
        self.assertIn("No significant rain", text)

    def test_rain_that_eases(self):
        rain = [5.0, 4.0, 3.0] + [0.0] * 21
        self.assertIn("ease in about 3 hour", flood_duration_agent.describe_rain(rain))

    def test_rain_that_continues(self):
        self.assertIn("still forecast at the end", flood_duration_agent.describe_rain([2.0] * 24))

    def test_high_tide_is_mentioned_and_a_low_one_is_not(self):
        self.assertIn("High tide", flood_duration_agent.describe_tide(1.5))
        self.assertEqual(flood_duration_agent.describe_tide(0.4), "")
        self.assertEqual(flood_duration_agent.describe_tide(None), "")

    def test_missing_rain_data_is_said(self):
        text = flood_duration_agent.format_duration("marikina", None, None, NOW)
        self.assertIn("could not read the rain forecast", text)

    def test_without_a_city_it_asks_and_waits(self):
        reply = flood_duration_agent.answer_duration_question("Gaano katagal ang baha?", ChatContext())
        self.assertEqual(reply.pending_intent, "duration")

    def test_the_answer_uses_the_forecast_and_the_tide(self):
        context = ChatContext(now=NOW)
        with mock.patch.object(flood_duration_agent, "download_rain", return_value={}), \
             mock.patch.object(flood_duration_agent, "split_rain", return_value=([], [3.0, 0.0] + [0.0] * 22)), \
             mock.patch.object(flood_duration_agent, "fetch_peak_tide", return_value=1.4):
            reply = flood_duration_agent.answer_duration_question("Gaano katagal ang baha sa Marikina?", context)
        self.assertIn("ease in about 1 hour", reply.text)
        self.assertIn("High tide", reply.text)


class PagasaFreshnessTests(unittest.TestCase):
    BODY = " Rainfall warning bulletin for Metro Manila." * 10

    def page(self, header):
        return {"text": header + self.BODY, "source": "https://www.pagasa.dost.gov.ph/weather"}

    def test_issue_time_formats(self):
        self.assertEqual(backup_sources.find_issue_time("Issued at 5:00 PM, 31 October 2026"), datetime(2026, 10, 31, 17, 0, tzinfo=PHILIPPINE_TZ))
        self.assertEqual(backup_sources.find_issue_time("as of October 29, 2026"), datetime(2026, 10, 29, 23, 59, tzinfo=PHILIPPINE_TZ))
        self.assertIsNone(backup_sources.find_issue_time("Typhoon Ondoy hit on 26 September 2009."))  # not an issue time
        self.assertIsNone(backup_sources.find_issue_time("Issued at 5:00 PM, 31 February 2026"))  # not a real date

    def test_a_bulletin_older_than_a_day_is_not_reliable(self):
        self.assertFalse(backup_sources.is_pagasa_reliable([self.page("Issued at 5:00 AM, 29 October 2026.")], NOW))
        self.assertEqual(
            backup_sources.pagasa_missing_reason([self.page("Issued at 5:00 AM, 29 October 2026.")], NOW),
            "PAGASA bulletin (out of date)",
        )

    def test_a_current_bulletin_is_reliable(self):
        self.assertTrue(backup_sources.is_pagasa_reliable([self.page("Issued at 5:00 PM, 31 October 2026.")], NOW))

    def test_a_bulletin_with_no_issue_time_cannot_be_judged_and_is_kept(self):
        self.assertTrue(backup_sources.is_pagasa_reliable([self.page("")], NOW))

    def test_a_fresh_page_makes_up_for_a_stale_one(self):
        pages = [self.page("Issued at 5:00 AM, 29 October 2026."), self.page("Issued at 5:00 PM, 31 October 2026.")]
        self.assertTrue(backup_sources.is_pagasa_reliable(pages, NOW))

    def test_the_cutoff_is_a_day(self):
        issued = NOW - timedelta(hours=23)
        header = f"Issued at {issued.strftime('%I:%M %p, %d %B %Y')}."
        self.assertTrue(backup_sources.is_pagasa_reliable([self.page(header)], NOW))


class SourceListTests(unittest.TestCase):
    CHUNKS = [
        Chunk("a", 0, "one", metadata={"source": "MMFMP.pdf", "agency": "MMDA", "issued_at": "2014-01-01T00:00:00", "page": 4}),
        Chunk("b", 0, "two", metadata={"source": "manual.pdf"}),
        Chunk("c", 0, "three", metadata={"source": "irr.pdf", "agency": "NDRRMC"}),
    ]

    def test_only_cited_sources_are_listed_with_their_labels(self):
        text = document_agents.format_source_list(self.CHUNKS, "Pack food [3]. Go to the barangay [1].")
        self.assertIn("[1] MMFMP.pdf, MMDA, 2014-01-01, page 4", text)
        self.assertIn("[3] irr.pdf, NDRRMC", text)
        self.assertNotIn("[2]", text)

    def test_no_citation_means_no_list(self):
        self.assertEqual(document_agents.format_source_list(self.CHUNKS, "I have no verified information."), "")

    def test_a_citation_number_that_does_not_exist_is_ignored(self):
        self.assertEqual(document_agents.format_source_list(self.CHUNKS, "See [9]."), "")


class KeptBarangayTests(unittest.TestCase):
    def test_a_typed_place_that_is_a_known_barangay_keeps_it(self):
        found = location_agent.resolve_location("Nasa Baclaran, Parañaque ako", geocoder=lambda query: (14.52, 120.99))
        self.assertEqual(found.barangay, "Baclaran")
        self.assertEqual(found.precision, "barangay")

    def test_an_unknown_subdivision_has_no_barangay(self):
        found = location_agent.resolve_location("Nasa Gentry Manor, Parañaque ako", geocoder=lambda query: (14.52, 120.99))
        self.assertIsNone(found.barangay)
        self.assertEqual(found.precision, "place")


if __name__ == "__main__":
    unittest.main()
