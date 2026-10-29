"""WebSocket endpoint that lets the browser stream microphone audio and get live transcripts.

The browser connects to this server, not to Meta, so the Meta API key never
leaves the backend:

    browser mic --PCM frames--> /ws/voice --PCM frames--> Meta speech-to-text
    browser     <--JSON events-- /ws/voice <--events------ Meta speech-to-text

Connect to ``/ws/voice?mode=endpointing&rate=24000``:

    mode  endpointing (default, hands-free, several questions per connection)
          or push_to_talk (one transcript; send {"type": "end"} when the button is released)
    rate  16000 or 24000, the sample rate of the audio the browser sends

Browser -> server:
    binary frames   raw 16-bit little-endian mono PCM at ``rate`` (about 80 ms each, in real time)
    {"type": "end"} no more audio; the server finishes and closes the socket

Server -> browser (JSON):
    {"type": "ready", "sessionId": ...}
    {"type": "partial", "text": ...}                 text so far; replaces the previous partial
    {"type": "final", "text": ..., "turnId": ...}    one finished question
    {"type": "error", "message": ...}

Add it to the FastAPI app with ``app.include_router(create_voice_router())``.
"""

import asyncio
import json
import logging
import os
from collections.abc import Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websockets.exceptions import WebSocketException

from voice.asr_realtime import AUDIO_ENCODINGS, RealtimeTranscriber, to_client_event

logger = logging.getLogger(__name__)

MAX_SESSION_SECONDS = 120  # one connection is cut off after this long, to cap speech-to-text cost
CLOSE_POLICY_VIOLATION = 1008
CLOSE_INTERNAL_ERROR = 1011
KEYWORDS = ["Marikina", "Pasig", "Taguig", "Valenzuela", "Las Piñas", "Quezon City", "PAGASA", "NDRRMC"]


def default_transcriber(mode: str, sample_rate: int) -> RealtimeTranscriber:
    """Build a transcriber using the MUSE_API key from the environment."""
    api_key = os.getenv("MUSE_API")
    if not api_key:
        raise RuntimeError("MUSE_API is not set")
    return RealtimeTranscriber(
        api_key,
        mode=mode,
        sample_rate=sample_rate,
        language_bias=["Tagalog", "English"],
        keywords=KEYWORDS,
    )


def create_voice_router(transcriber_factory: Callable[[str, int], RealtimeTranscriber] = default_transcriber) -> APIRouter:
    """Build the router. Tests pass their own ``transcriber_factory`` to avoid real calls."""
    router = APIRouter()

    @router.websocket("/ws/voice")
    async def voice_socket(websocket: WebSocket, mode: str = "endpointing", rate: int = 24000) -> None:
        await websocket.accept()

        if mode not in ("endpointing", "push_to_talk") or rate not in AUDIO_ENCODINGS:
            await websocket.close(code=CLOSE_POLICY_VIOLATION, reason="Bad mode or rate")
            return

        try:
            async with asyncio.timeout(MAX_SESSION_SECONDS):
                transcriber = transcriber_factory(mode.upper(), rate)
                async with transcriber:
                    await websocket.send_json({"type": "ready", "sessionId": transcriber.session_id})
                    await relay(websocket, transcriber)
            await websocket.close()
        except TimeoutError:
            await send_error_and_close(websocket, "Session time limit reached")
        except (RuntimeError, WebSocketException, OSError) as error:
            logger.error("Voice session failed: %s", error)
            await send_error_and_close(websocket, "Speech-to-text is unavailable right now")
        except WebSocketDisconnect:
            pass  # the browser left; nothing to report

    return router


async def relay(websocket: WebSocket, transcriber: RealtimeTranscriber) -> None:
    """Pass audio from the browser to Meta and Meta's events back, until the session ends."""

    async def audio_to_meta() -> None:
        """Forward microphone frames. Always ends the stream so Meta finishes and closes."""
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    await transcriber.send_audio(message["bytes"])
                elif message.get("text") and json.loads(message["text"]).get("type") == "end":
                    break
        finally:
            try:
                await transcriber.end_stream()
            except WebSocketException:
                pass  # Meta already closed the socket

    upload = asyncio.create_task(audio_to_meta())
    try:
        # Meta closes the socket once it has sent everything after "endStream".
        async for event in transcriber.events():
            client_event = to_client_event(event)
            if client_event:
                await websocket.send_json(client_event)
    finally:
        upload.cancel()


async def send_error_and_close(websocket: WebSocket, message: str) -> None:
    """Tell the browser what went wrong, then close. Ignores a socket that is already closed."""
    try:
        await websocket.send_json({"type": "error", "message": message})
        await websocket.close(code=CLOSE_INTERNAL_ERROR)
    except RuntimeError:
        pass
