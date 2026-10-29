"""WebSocket endpoint for a webcam flood check: the browser sends a picture every few seconds, SAM looks for flood water.

SAM works on pictures and clips, not on a live stream, so "live" here means a frame every few
seconds. The browser grabs the frame from the webcam (``getUserMedia``), shrinks it and sends it as a
JPEG; the server waits for each answer before the browser sends the next one.

Connect to ``/ws/camera`` (add ``?people=1`` to also count people and vehicles; that is 3 SAM
requests per frame instead of 1).

Browser -> server:  binary JPEG frames (at most about 1 MB each; 640 pixels wide is plenty)
Server -> browser (JSON):
    {"type": "checking"}                              a frame was accepted
    {"type": "report", "report": {...}, "text": ...}  the result (see FloodSceneReport) and the text for the user
    {"type": "skipped", "wait_seconds": ...}          a frame came too soon and was ignored
    {"type": "error", "message": ...}                 one frame failed; the socket stays open

Cost limits: at most one check every ``MIN_SECONDS_BETWEEN_CHECKS`` seconds, and a connection
closes after ``MAX_SESSION_SECONDS``. Add it with ``app.include_router(create_camera_router())``.
"""

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agents.flood_media_agent import format_report
from agents.flood_types import ChatContext
from video.flood_scene import FloodSceneReport, analyze_flood_media_async
from video.sam_client import ImageInput, SamError

logger = logging.getLogger(__name__)

MIN_SECONDS_BETWEEN_CHECKS = 3.0
MAX_SESSION_SECONDS = 300
MAX_FRAME_BYTES = 1_000_000
CLOSE_INTERNAL_ERROR = 1011

Analyzer = Callable[..., Awaitable[FloodSceneReport]]


def create_camera_router(analyze: Analyzer = analyze_flood_media_async) -> APIRouter:
    """Build the router. Tests pass their own ``analyze`` function to avoid real SAM calls."""
    router = APIRouter()

    @router.websocket("/ws/camera")
    async def camera_socket(websocket: WebSocket, people: bool = False) -> None:
        await websocket.accept()
        api_key = os.getenv("MUSE_API")
        if not api_key:
            await send_error_and_close(websocket, "The picture check is unavailable right now")
            return

        last_check_started = float("-inf")
        try:
            async with asyncio.timeout(MAX_SESSION_SECONDS):
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    frame = message.get("bytes")
                    if not frame:
                        continue

                    wait = MIN_SECONDS_BETWEEN_CHECKS - (time.monotonic() - last_check_started)
                    if wait > 0:
                        await websocket.send_json({"type": "skipped", "wait_seconds": round(wait, 1)})
                        continue
                    if len(frame) > MAX_FRAME_BYTES:
                        await websocket.send_json({"type": "error", "message": "Frame is too large"})
                        continue

                    last_check_started = time.monotonic()
                    await websocket.send_json({"type": "checking"})
                    await check_frame(websocket, analyze, frame, api_key, people)
        except TimeoutError:
            await send_error_and_close(websocket, "Session time limit reached")
        except WebSocketDisconnect:
            pass  # the browser left

    return router


async def check_frame(websocket: WebSocket, analyze: Analyzer, frame: bytes, api_key: str, people: bool) -> None:
    """Analyse one frame and send the result. A failed frame is reported; the session goes on."""
    try:
        report = await analyze(ImageInput(data=frame), api_key, include_people_and_vehicles=people)
    except (SamError, OSError, ValueError) as error:
        logger.error("Webcam frame check failed: %s", error)
        await websocket.send_json({"type": "error", "message": "Could not check this frame"})
        return
    await websocket.send_json({"type": "report", "report": report.to_dict(), "text": format_report(report, ChatContext())})


async def send_error_and_close(websocket: WebSocket, message: str) -> None:
    """Tell the browser what went wrong, then close. Ignores a socket that is already closed."""
    try:
        await websocket.send_json({"type": "error", "message": message})
        await websocket.close(code=CLOSE_INTERNAL_ERROR)
    except RuntimeError:
        pass
