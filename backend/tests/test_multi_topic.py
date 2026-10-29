"""Tests for messages with several topics: the splitter and the multi node. No network, database or model."""

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

from agents.flood_types import AgentReply
from langgraph_orchestrator import graph as chat_graph
from langgraph_orchestrator.intents import (
    EVACUATION, FLOOD_RISK, PREPAREDNESS, WEATHER, classify_intent, split_by_intent,
)

USER_MESSAGE = (
    "Icheck mo ang current location ko ngayon. Mag forecast ka ng 7 days at kung bumaha hanggang tuhod dito ngayon, "
    "ano ang kailangan kong gawin at saan ang pinaka safe na evacuation center. Malapit kasi kami sa dagat."
)


class SplitTests(unittest.TestCase):
    def test_three_topics_in_answer_order(self):
        parts = split_by_intent(USER_MESSAGE)
        self.assertEqual([intent for intent, _ in parts], [WEATHER, PREPAREDNESS, EVACUATION])

    def test_context_reaches_the_questions_that_need_it(self):
        questions = dict(split_by_intent(USER_MESSAGE))
        self.assertIn("7 days", questions[WEATHER])
        self.assertNotIn("tuhod", questions[WEATHER])  # the flood condition is not about the forecast
        self.assertIn("tuhod", questions[PREPAREDNESS])
        self.assertIn("tuhod", questions[EVACUATION])
        self.assertIn("dagat", questions[EVACUATION])  # a part with no topic goes to every question

    def test_single_topic_messages_are_not_split(self):
        for text in ["ano ang gagawin kapag bumaha", "evacuation center sa Pasig", "Uulan ba bukas?"]:
            self.assertEqual(split_by_intent(text), [], text)

    def test_emergency_is_never_split(self):
        self.assertEqual(split_by_intent("trapped kami, may forecast ba?"), [])

    def test_flood_risk_counts_when_it_is_the_only_flood_topic(self):
        parts = split_by_intent("May bagyo ba? baha ba sa Marikina?")
        self.assertEqual([intent for intent, _ in parts], [WEATHER, FLOOD_RISK])

    def test_new_preparedness_phrases(self):
        self.assertEqual(classify_intent("ano ang kailangan kong gawin"), PREPAREDNESS)


class MultiNodeTests(unittest.TestCase):
    """The real graph, with the agents replaced by fakes that record the question they were given."""

    def setUp(self):
        self.asked = {}

    def fake_agents(self):
        def weather(question, llm):
            self.asked["weather"] = question
            return AgentReply(text="FORECAST")

        def documents(question, llm, retriever):
            self.asked["documents"] = question
            return AgentReply(text="STEPS")

        def evacuation(question, context):
            self.asked["evacuation"] = (question, context.location)
            return AgentReply(text="CENTER")

        return mock.patch.multiple(
            chat_graph,
            answer_weather_question=weather,
            answer_document_question=documents,
            answer_evacuation_question=evacuation,
        )

    def ask(self, graph, text, thread, location=None):
        return asyncio.run(chat_graph.ask_reply(graph, text, thread, location=location))

    def build(self):
        # A retriever object is needed so the documents node calls the (fake) document agent.
        return chat_graph.build_graph(FakeListChatModel(responses=["x"]), retriever=object(), checkpointer=MemorySaver())

    def test_all_topics_answered_when_the_location_is_known(self):
        with self.fake_agents():
            reply = self.ask(self.build(), USER_MESSAGE, "t1", location={"lat": 14.65, "lng": 121.10})
        for text in ("FORECAST", "STEPS", "CENTER"):
            self.assertIn(text, reply.text)
        self.assertLess(reply.text.index("FORECAST"), reply.text.index("STEPS"))
        self.assertLess(reply.text.index("STEPS"), reply.text.index("CENTER"))
        self.assertNotIn("Which city", reply.text)

    def test_weather_question_gets_the_users_city(self):
        with self.fake_agents():
            self.ask(self.build(), USER_MESSAGE, "t1", location={"lat": 14.65, "lng": 121.10})
        self.assertIn("(marikina)", self.asked["weather"])

    def test_topics_needing_a_location_wait_for_the_city(self):
        with self.fake_agents():
            graph = self.build()  # built inside the patch, because the graph keeps the agents it is built with
            first = self.ask(graph, USER_MESSAGE, "t2")
            self.assertIn("FORECAST", first.text)  # the topics that need no location are answered now
            self.assertIn("STEPS", first.text)
            self.assertNotIn("CENTER", first.text)
            self.assertIn("Which city", first.text)

            second = self.ask(graph, "Marikina", "t2")  # just the city finishes the waiting topic
        self.assertIn("CENTER", second.text)
        self.assertNotIn("FORECAST", second.text)  # and does not repeat the earlier answers
        self.assertEqual(self.asked["evacuation"][1].city, "marikina")

    def test_one_failing_agent_does_not_lose_the_others(self):
        with self.fake_agents(), mock.patch.object(chat_graph, "answer_evacuation_question", side_effect=RuntimeError("boom")):
            reply = self.ask(self.build(), USER_MESSAGE, "t3", location={"lat": 14.65, "lng": 121.10})
        self.assertIn("FORECAST", reply.text)
        self.assertIn("STEPS", reply.text)
        self.assertNotIn("CENTER", reply.text)

    def test_safe_answer_flag_is_kept(self):
        def safe_documents(question, llm, retriever):
            return AgentReply(text="SAFE", safe_answer=True)

        with self.fake_agents(), mock.patch.object(chat_graph, "answer_document_question", safe_documents):
            reply = self.ask(self.build(), USER_MESSAGE, "t4", location={"lat": 14.65, "lng": 121.10})
        self.assertTrue(reply.safe_answer)


if __name__ == "__main__":
    unittest.main()
