"""Tests for how the chat graph routes messages and remembers the conversation. No network or database."""

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("bm25s", types.ModuleType("bm25s"))

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver

from agents import evacuation_agent, route_agent
from agents.flood_types import AgentReply
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


class MediaTests(unittest.TestCase):
    """A photo or video attached to a message is checked once, by the flood media node."""

    def setUp(self):
        self.upload_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.upload_dir.cleanup)
        env = mock.patch.dict("os.environ", {chat_graph.MEDIA_UPLOAD_DIR_VARIABLE: self.upload_dir.name})
        env.start()
        self.addCleanup(env.stop)

    def route(self, text, media=None):
        message = chat_graph.add_messages([], [("user", text)])
        return chat_graph.route_message({"messages": message, "media": media})

    def test_media_goes_to_the_media_node_but_an_emergency_still_wins(self):
        media = {"kind": "image", "url": "https://example.com/a.jpg"}
        self.assertEqual(self.route("look", media), chat_graph.FLOOD_MEDIA_NODE)
        self.assertEqual(self.route("trapped, see photo", media), chat_graph.EMERGENCY_NODE)
        self.assertEqual(self.route("look"), chat_graph.DOCUMENTS_NODE)

    def test_media_references_are_checked(self):
        photo = Path(self.upload_dir.name) / "street.jpg"
        photo.write_bytes(b"jpeg")
        self.assertEqual(chat_graph.media_from_dict({"kind": "image", "path": str(photo)}).path, photo.resolve())
        self.assertEqual(chat_graph.media_from_dict({"kind": "video", "url": "https://example.com/v.mp4"}).url,
                         "https://example.com/v.mp4")

        bad = [
            {"kind": "audio", "url": "https://example.com/a"},
            {"kind": "image"},
            {"kind": "image", "url": "https://example.com/a", "path": str(photo)},
            {"kind": "image", "url": "http://example.com/a.jpg"},       # not https
            {"kind": "image", "url": "https://localhost/a.jpg"},
            {"kind": "image", "url": "https://127.0.0.1/a.jpg"},
            {"kind": "image", "url": "https://192.168.1.5/a.jpg"},
            {"kind": "image", "path": "/etc/passwd"},                    # outside the upload folder
            {"kind": "image", "path": str(Path(self.upload_dir.name) / "missing.jpg")},
        ]
        for media in bad:
            with self.assertRaises(ValueError, msg=media):
                chat_graph.media_from_dict(media)

    def test_the_graph_checks_the_media_once_and_passes_the_depth_option(self):
        seen = {}

        def fake_agent(media, context, estimate_depth=False):
            seen["media"], seen["depth"] = media, estimate_depth
            return AgentReply(text="Flood water covers a large part of the picture.")

        graph = chat_graph.build_graph(FakeListChatModel(responses=["unused"]), checkpointer=MemorySaver())

        async def talk():
            with mock.patch.object(chat_graph, "answer_flood_media_question", fake_agent):
                first = await chat_graph.ask(
                    graph, "", "thread-media",
                    media={"kind": "image", "url": "https://example.com/a.jpg", "estimate_depth": True},
                )
                second = await chat_graph.ask(graph, "hello", "thread-media")  # no media: must not be checked again
            return first, second

        first, second = asyncio.run(talk())
        self.assertIn("Flood water covers", first)
        self.assertEqual(seen["media"].url, "https://example.com/a.jpg")
        self.assertTrue(seen["depth"])
        self.assertNotIn("Flood water covers", second)

    def test_unusable_media_gets_a_polite_reply(self):
        graph = chat_graph.build_graph(FakeListChatModel(responses=["unused"]))
        reply = asyncio.run(chat_graph.ask(graph, "see this", "thread-bad", media={"kind": "image", "path": "/etc/passwd"}))
        self.assertEqual(reply, chat_graph.BAD_MEDIA_ANSWER)


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
