"""Tests for the LangSmith helpers: nothing private may be sent."""

from agents.flood_types import AgentReply, ChatContext, Location
from utils.tracing import clean_inputs, clean_outputs, enable_langsmith, redact


def test_phone_numbers_are_replaced_but_normal_text_is_kept():
    assert redact("Call 0917 123 4567 or +63 2 8888 8999 now") == "Call [number] or [number] now"
    assert redact("Evacuate to Barangay 12 on 3rd street, 911 is free") == "Evacuate to Barangay 12 on 3rd street, 911 is free"


def test_secrets_and_services_are_never_sent():
    place = Location(name="Marikina", lat=14.65, lng=121.10, city="marikina")
    inputs = clean_inputs(
        {"question": "hotline 09171234567", "api_key": "abc123", "context": ChatContext(location=place), "llm": object()}
    )
    assert inputs["api_key"] == "[hidden]"
    assert inputs["question"] == "hotline [number]"
    assert inputs["context"] == {"location": {"city": "marikina", "precision": "city"}}  # no coordinates
    assert inputs["llm"] == "<object>"


def test_outputs_are_cleaned():
    assert clean_outputs(AgentReply(text="Call 09171234567")) == {"text": "Call [number]", "pending_intent": None}
    assert clean_outputs("plain") == {"output": "plain"}


def test_tracing_needs_both_the_switch_and_a_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACE", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_KEY", raising=False)
    monkeypatch.setattr("utils.tracing.load_dotenv", lambda: None)  # ignore the real .env
    # enable_langsmith writes these two; setting them here first lets pytest restore them afterwards.
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    assert enable_langsmith() is False

    monkeypatch.setenv("LANGSMITH_KEY", "k")
    assert enable_langsmith() is True
    import os
    assert os.environ["LANGSMITH_API_KEY"] == "k" and os.environ["LANGSMITH_TRACING"] == "true"
    monkeypatch.setenv("LANGSMITH_TRACE", "false")
    assert enable_langsmith() is False


def test_sanitize_cleans_nested_run_data():
    from utils.tracing import sanitize

    run_input = {
        "messages": [{"content": "my number is 09171234567"}],
        "location": {"lat": 14.650712, "lng": 121.102934, "city": "marikina"},
        "api_key": "abc123",
    }
    assert sanitize(run_input) == {
        "messages": [{"content": "my number is [number]"}],
        "location": {"lat": 14.65, "lng": 121.1, "city": "marikina"},
        "api_key": "[hidden]",
    }


def test_sanitize_cleans_langchain_messages_and_dataclasses():
    from langchain_core.messages import HumanMessage

    from utils.tracing import sanitize

    cleaned = sanitize({"messages": [HumanMessage(content="call 09171234567")], "place": Location("Here", 14.650712, 121.1029, "marikina")})
    assert cleaned["messages"][0]["content"] == "call [number]"
    assert cleaned["place"]["lat"] == 14.65
