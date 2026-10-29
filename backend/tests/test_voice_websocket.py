"""Tests for the voice WebSocket endpoint, using a fake speech-to-text session (no network)."""

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from voice.asr_realtime import to_client_event
from voice.websocket_router import create_voice_router


class FakeTranscriber:
    """Stands in for ``RealtimeTranscriber``: counts audio, answers once the stream ends."""

    def __init__(self, mode, sample_rate):
        self.mode, self.sample_rate = mode, sample_rate
        self.session_id = "fake-session"
        self.audio_bytes = 0
        self._ended = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def send_audio(self, pcm_bytes):
        self.audio_bytes += len(pcm_bytes)

    async def end_stream(self):
        self._ended.set()

    async def events(self):
        await self._ended.wait()
        yield {"type": "speechStart", "turnId": 1}
        yield {"type": "transcript", "transcript": "will it", "final": False}
        yield {"type": "speechComplete", "turnId": 1, "transcript": f"heard {self.audio_bytes} bytes"}


def make_client(factory=FakeTranscriber) -> TestClient:
    app = FastAPI()
    app.include_router(create_voice_router(factory))
    return TestClient(app)


def test_audio_in_transcripts_out():
    with make_client().websocket_connect("/ws/voice?mode=endpointing&rate=16000") as socket:
        assert socket.receive_json() == {"type": "ready", "sessionId": "fake-session"}
        socket.send_bytes(b"\x00" * 100)
        socket.send_bytes(b"\x00" * 50)
        socket.send_json({"type": "end"})

        assert socket.receive_json() == {"type": "partial", "text": "will it"}
        assert socket.receive_json() == {"type": "final", "text": "heard 150 bytes", "turnId": 1}


def test_bad_mode_is_rejected():
    with make_client().websocket_connect("/ws/voice?mode=shouting") as socket:
        message = socket.receive()
        assert message["type"] == "websocket.close"
        assert message["code"] == 1008


def test_missing_key_reports_an_error():
    def no_key(mode, sample_rate):
        raise RuntimeError("MUSE_API is not set")

    with make_client(no_key).websocket_connect("/ws/voice") as socket:
        assert socket.receive_json() == {"type": "error", "message": "Speech-to-text is unavailable right now"}


def test_event_mapping():
    assert to_client_event({"type": "transcript", "transcript": "ha", "final": False}) == {"type": "partial", "text": "ha"}
    assert to_client_event({"type": "transcript", "transcript": "hello", "final": True}) == {"type": "final", "text": "hello"}
    assert to_client_event({"type": "speechStart", "turnId": 1}) is None
    assert to_client_event({"type": "error", "message": "boom"}) == {"type": "error", "message": "boom"}
