"""Tests for the low-confidence web fallback. No network, browser or model is used.

Run from the backend folder:  python -m unittest discover -s tests -v
"""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("bm25s", types.ModuleType("bm25s"))  # optional search library, not needed here

from agents import document_agents, flood_risk_agent, weather_agent, web_lookup_agent
from agents.flood_types import ChatContext, Location
from agents.safe_answer import SAFE_ANSWER
from src.rag.models import Chunk

WEB_PAGE = {"page": "web", "source": "https://ndrrmc.gov.ph/update", "title": "Update", "text": "Flood update text"}


def chunk_with_score(score: float) -> Chunk:
    return Chunk(doc_id="doc", chunk_idx=0, content="some text", score=score, metadata={})


class FakeLLM:
    """Stands in for the chat model; web_lookup_agent only needs it inside a chain, which is patched out."""


class TrustedUrlTests(unittest.TestCase):
    def test_trusted_domains_and_subdomains(self):
        self.assertTrue(web_lookup_agent.is_trusted_url("https://ndrrmc.gov.ph/news"))
        self.assertTrue(web_lookup_agent.is_trusted_url("https://www.dswd.gov.ph/news"))

    def test_other_hosts_and_plain_http_are_rejected(self):
        self.assertFalse(web_lookup_agent.is_trusted_url("https://evil.example.com/ndrrmc.gov.ph"))
        self.assertFalse(web_lookup_agent.is_trusted_url("https://notndrrmc.gov.ph/"))
        self.assertFalse(web_lookup_agent.is_trusted_url("http://ndrrmc.gov.ph/news"))

    def test_result_links_stay_on_the_site(self):
        html = (
            '<a href="/news/1">one</a><a href="https://evil.example.com/x">bad</a>'
            '<a href="/news/1">again</a><a href="/news/2#top">two</a><a href="/news/3">three</a>'
        )
        links = web_lookup_agent.find_result_links(html, "https://ndrrmc.gov.ph/?s=flood", "ndrrmc.gov.ph")
        self.assertEqual(links, ["https://ndrrmc.gov.ph/news/1", "https://ndrrmc.gov.ph/news/2"])


class DocumentFallbackTests(unittest.TestCase):
    def setUp(self):
        document_agents.answer_cache = type(document_agents.answer_cache)(ttl_seconds=60)

    def test_confidence_check(self):
        self.assertTrue(document_agents.is_low_confidence([]))
        self.assertTrue(document_agents.is_low_confidence([chunk_with_score(0.1)]))
        self.assertFalse(document_agents.is_low_confidence([chunk_with_score(0.1), chunk_with_score(0.9)]))

    def test_weak_match_uses_the_web(self):
        with mock.patch.object(document_agents, "find_sources", return_value=[chunk_with_score(0.1)]), \
             mock.patch.object(document_agents, "answer_from_web", return_value="web answer") as web:
            answer = document_agents.answer_document_question("question", FakeLLM(), retriever=None)
        self.assertEqual(answer.text, "web answer")
        self.assertFalse(answer.safe_answer)
        web.assert_called_once()

    def test_very_low_confidence_skips_the_web_and_gives_the_safe_answer(self):
        self.assertTrue(document_agents.is_very_low_confidence([chunk_with_score(0.05)]))
        self.assertFalse(document_agents.is_very_low_confidence([]))  # nothing found: the web is still tried
        with mock.patch.object(document_agents, "find_sources", return_value=[chunk_with_score(0.05)]), \
             mock.patch.object(document_agents, "answer_from_web") as web:
            answer = document_agents.answer_document_question("question", FakeLLM(), retriever=None)
        web.assert_not_called()
        self.assertEqual(answer.text, SAFE_ANSWER)
        self.assertTrue(answer.safe_answer)

    def test_nothing_on_the_web_gives_the_fixed_reply(self):
        with mock.patch.object(document_agents, "find_sources", return_value=[]), \
             mock.patch.object(document_agents, "answer_from_web", return_value=None):
            answer = document_agents.answer_document_question("question", FakeLLM(), retriever=None)
        self.assertEqual(answer.text, SAFE_ANSWER)
        self.assertTrue(answer.safe_answer)

    def test_confident_match_does_not_use_the_web(self):
        with mock.patch.object(document_agents, "find_sources", return_value=[chunk_with_score(0.9)]), \
             mock.patch.object(document_agents, "answer_from_web") as web, \
             mock.patch.object(document_agents, "model_budget") as budget:
            budget.try_use.return_value = False  # stop right after the confidence check
            document_agents.answer_document_question("question", FakeLLM(), retriever=None)
        web.assert_not_called()


class WeatherFallbackTests(unittest.TestCase):
    def setUp(self):
        weather_agent.answer_cache = type(weather_agent.answer_cache)(ttl_seconds=60)

    def test_no_pagasa_and_no_web_and_no_forecast_gives_the_fixed_reply(self):
        with mock.patch.object(weather_agent, "fetch_both_sources", return_value=([], None)), \
             mock.patch.object(weather_agent, "search_trusted_sites", return_value=[]):
            reply = weather_agent.answer_weather_question("ulan?", FakeLLM())
            self.assertEqual(reply.text, SAFE_ANSWER)
            self.assertTrue(reply.safe_answer)

    def test_pagasa_present_skips_the_web(self):
        with mock.patch.object(weather_agent, "fetch_both_sources", return_value=([WEB_PAGE], None)), \
             mock.patch.object(weather_agent, "search_trusted_sites") as web, \
             mock.patch.object(weather_agent, "model_budget") as budget:
            budget.try_use.return_value = False
            weather_agent.answer_weather_question("ulan?", FakeLLM())
        web.assert_not_called()


class FloodRiskFallbackTests(unittest.TestCase):
    def test_web_links_are_added_but_do_not_change_the_level(self):
        context = ChatContext(location=Location("Marikina", 14.65, 121.10, "marikina"))
        with mock.patch("agents.fetch_weather_using_browser.fetch_weather_pages", return_value=[]), \
             mock.patch.object(flood_risk_agent, "search_trusted_sites", return_value=[WEB_PAGE]), \
             mock.patch.object(flood_risk_agent, "assess_flood_risk") as assess:
            assess.return_value = flood_risk_agent.FloodRisk(city="marikina", level="low")
            reply = flood_risk_agent.answer_flood_risk_question("baha ba?", context)
        self.assertIn("https://ndrrmc.gov.ph/update", reply.text)
        self.assertIn("LOW", reply.text)


if __name__ == "__main__":
    unittest.main()
