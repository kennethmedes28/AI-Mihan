"""Tests for the webcam WebSocket, using a fake analyzer (no SAM calls)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from video import camera_router
from video.camera_router import create_camera_router
from video.flood_scene import LEVEL_SOME, FloodSceneReport
from video.sam_client import SamError


def fake_report() -> FloodSceneReport:
    return FloodSceneReport("image", 1, 0.1, 0.1, 1.0, LEVEL_SOME, 0, 0, 0, 0)


def make_client(analyze) -> TestClient:
    app = FastAPI()
    app.include_router(create_camera_router(analyze))
    return TestClient(app)


def test_frame_in_report_out(monkeypatch):
    monkeypatch.setenv("MUSE_API", "key")
    seen = {}

    async def analyze(media, api_key, include_people_and_vehicles, estimate_depth=False):
        seen["bytes"], seen["people"] = media.data, include_people_and_vehicles
        return fake_report()

    with make_client(analyze).websocket_connect("/ws/camera?people=1") as socket:
        socket.send_bytes(b"jpeg-bytes")
        assert socket.receive_json() == {"type": "checking"}
        message = socket.receive_json()
        assert message["type"] == "report" and message["report"]["level"] == LEVEL_SOME
        assert "Flood water covers" in message["text"]
    assert seen == {"bytes": b"jpeg-bytes", "people": True}


def test_frames_that_come_too_soon_are_skipped(monkeypatch):
    monkeypatch.setenv("MUSE_API", "key")

    async def analyze(media, api_key, include_people_and_vehicles, estimate_depth=False):
        return fake_report()

    with make_client(analyze).websocket_connect("/ws/camera") as socket:
        socket.send_bytes(b"one")
        socket.receive_json(), socket.receive_json()  # checking, report
        socket.send_bytes(b"two")
        assert socket.receive_json()["type"] == "skipped"


def test_a_failed_frame_does_not_close_the_session(monkeypatch):
    monkeypatch.setenv("MUSE_API", "key")

    async def analyze(media, api_key, include_people_and_vehicles, estimate_depth=False):
        raise SamError("boom")

    with make_client(analyze).websocket_connect("/ws/camera") as socket:
        socket.send_bytes(b"one")
        socket.receive_json()  # checking
        assert socket.receive_json() == {"type": "error", "message": "Could not check this frame"}


def test_oversized_frame_and_missing_key(monkeypatch):
    monkeypatch.setenv("MUSE_API", "key")
    monkeypatch.setattr(camera_router, "MAX_FRAME_BYTES", 5)

    async def analyze(media, api_key, include_people_and_vehicles, estimate_depth=False):
        return fake_report()

    with make_client(analyze).websocket_connect("/ws/camera") as socket:
        socket.send_bytes(b"too-big-frame")
        assert socket.receive_json()["message"] == "Frame is too large"

    monkeypatch.delenv("MUSE_API")
    with make_client(analyze).websocket_connect("/ws/camera") as socket:
        assert socket.receive_json()["type"] == "error"
