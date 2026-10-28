"""Tests for how the chat graph routes messages and remembers the conversation. No network or database."""

import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("bm25s", types.ModuleType("bm25s"))

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver

from agents import evacuation_agent, route_agent
from langgraph_orchestrator import graph as chat_graph

CENTERS = [{"name": "Marikina Sports Center", "lat": 14.649, "lng": 121.101, "city": "Marikina", "capacity": 500}]


class RoutingTests(unittest.TestCase):
    def route(self, text, pending=None):
        message = chat_graph.add_messages([], [("user", text)])
        return chat_graph.route_message({"messages": message, "pending_intent": pending})

    def test_emergency_beats_everything(self):
        self.assertEqual(self.route("trapped in Marikina", pending="route"), chat_graph.EMERGENCY_NODE)

    def test_bare_place_answers_the_waiting_agent(self):
        self.assertEqual(self.route("Marikina", pending="evacuation"), chat_graph.EVACUATION_NODE)

    def test_bare_place_without_waiting_agent_sets_location(self):
        self.assertEqual(self.route("Marikina"), chat_graph.LOCATION_NODE)

    def test_new_topic_is_not_hijacked_by_waiting_agent(self):
        self.assertEqual(self.route("hotline Pasig", pending="evacuation"), chat_graph.CONTACTS_NODE)

    def test_unknown_text_goes_to_documents(self):
        self.assertEqual(self.route("hello"), chat_graph.DOCUMENTS_NODE)


class ConversationTests(unittest.TestCase):
    def test_location_and_destination_are_remembered(self):
        risk = mock.Mock(level="low")
        with mock.patch.object(evacuation_agent, "fetch_evacuation_centers", return_value=CENTERS), \
             mock.patch.object(evacuation_agent, "assess_flood_risk", return_value=risk), \
             mock.patch.object(route_agent, "assess_flood_risk", return_value=risk), \
             mock.patch.dict("os.environ", {}, clear=True):
            graph = chat_graph.build_graph(
                FakeListChatModel(responses=["unused"]), checkpointer=MemorySaver(), get_connection=lambda: object()
            )

            async def talk():
                replies = []
                for text in ["nearest evacuation center?", "Marikina", "how do I get there?"]:
                    replies.append(await chat_graph.ask(graph, text, "thread-1"))
                return replies

            asked, found, routed = asyncio.run(talk())

        self.assertIn("Which city", asked)  # no location yet, so it asks
        self.assertIn("Marikina Sports Center", found)  # the answer went back to the evacuation agent
        self.assertIn("Marikina Sports Center", routed)  # "there" meant the remembered destination

    def test_agent_crash_gives_polite_reply(self):
        with mock.patch.object(chat_graph, "answer_contacts_question", side_effect=RuntimeError("boom")):
            graph = chat_graph.build_graph(FakeListChatModel(responses=["unused"]))
            reply = asyncio.run(chat_graph.ask(graph, "hotline Marikina", "thread-2"))
        self.assertEqual(reply, chat_graph.ERROR_ANSWER)

    def test_gps_fix_from_the_app_sets_the_city(self):
        with mock.patch.object(evacuation_agent, "fetch_evacuation_centers", return_value=CENTERS), \
             mock.patch.object(evacuation_agent, "assess_flood_risk", return_value=mock.Mock(level="low")):
            graph = chat_graph.build_graph(FakeListChatModel(responses=["unused"]), get_connection=lambda: object())
            reply = asyncio.run(
                chat_graph.ask(graph, "nearest evacuation center?", "t3", location={"lat": 14.6507, "lng": 121.1029})
            )
        self.assertIn("Marikina Sports Center", reply)


if __name__ == "__main__":
    unittest.main()
